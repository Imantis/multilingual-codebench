import contextlib
import email.utils
import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PAID_MAX_ATTEMPTS = 5
PAID_MAX_RETRY_WAIT_SECONDS = 900

class APIError(RuntimeError):
    def __init__(self, message, *, retry=None):
        super().__init__(message)
        self.retry = retry

def sampling_settings(provider, model):
    if model in ('openai/gpt-oss-20b', 'gpt-oss-20b'):
        if provider == 'groq':
            return {'temperature': 0.0, 'reasoning_effort': 'low'}
        if provider == 'openrouter':
            return {'temperature': 0.0, 'reasoning': {'effort': 'low'}}
    if provider=='openrouter' and model in ('openai/gpt-6-luna','anthropic/claude-haiku-5.5'):

        return {'reasoning':{'effort':'low'}}
    if provider=='openrouter' and model.removeprefix('google/').startswith('gemini-3'):
        return {'temperature':1.0,'reasoning':{'effort':'minimal' if 'lite' in model else 'low'}}
    if provider=='gemini' and model.startswith('gemini-3'):

        return {'thinking_level':'minimal' if 'lite' in model else 'low'}

    if model.removeprefix('google/').startswith('gemma-4-'):
        return {'temperature': 1.0, 'top_p': .95, 'top_k': 64}
    return {'temperature': 0.0}

class RateLimiter:
    def __init__(self, provider, key, rpm=10, path=None, clock=time.time, sleep=time.sleep):
        self.path = Path(path or os.getenv('CODEBENCH_RATE_DB', str(Path.home()/'.cache'/'multilingual-codebench'/'rate.sqlite')))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = hashlib.sha256((provider+':'+key).encode()).hexdigest()
        self.interval = 60/rpm + .1 if rpm else 0
        self.clock, self.sleep = clock, sleep
        with contextlib.closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS calls (identity TEXT PRIMARY KEY, next REAL)')
    def acquire(self):
        while True:
            with contextlib.closing(sqlite3.connect(self.path, timeout=60)) as db, db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT next FROM calls WHERE identity=?', (self.identity,)).fetchone()
                now = self.clock()
                delay = max(0, (row[0] if row else 0)-now)
                if not delay:
                    db.execute('INSERT OR REPLACE INTO calls VALUES (?,?)', (self.identity, now+self.interval))
                    return
            self.sleep(min(delay, 30))
    def defer(self, seconds):
        with contextlib.closing(sqlite3.connect(self.path, timeout=60)) as db, db:
            db.execute('INSERT INTO calls VALUES (?,?) ON CONFLICT(identity) DO UPDATE SET next=MAX(next,excluded.next)',
                       (self.identity, self.clock()+seconds))

def retry_after(value):
    if not value: return 0
    try: return max(0, float(value))
    except ValueError:
        try: return max(0, email.utils.parsedate_to_datetime(value).timestamp()-time.time())
        except (ValueError, TypeError): return 0


def reset_seconds(value):

    if not value: return 0
    try: return max(0, float(value))
    except ValueError:
        units = {'ms': .001, 's': 1, 'm': 60, 'h': 3600}
        return sum(float(number)*units[unit] for number,unit in
                   re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h)', value))

class PaidRequestBudget:

    def __init__(self, path, limit=None):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with contextlib.closing(sqlite3.connect(self.path,timeout=30)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, request_limit INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, started_at REAL, model TEXT, status TEXT, response_id TEXT, reported_usd REAL)')
            previous=db.execute('SELECT request_limit FROM budget WHERE id=1').fetchone()
            if previous and limit is not None and previous[0]!=limit:
                raise APIError('Paid request ceiling changed; retain the existing ceiling for this workdir')
            if not previous:
                if limit is None or limit<1: raise APIError('A positive paid request ceiling is required')
                db.execute('INSERT INTO budget VALUES (1,?)',(limit,))

    def reserve(self, model):
        with contextlib.closing(sqlite3.connect(self.path,timeout=30)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            limit=db.execute('SELECT request_limit FROM budget WHERE id=1').fetchone()[0]
            count=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
            if count>=limit:
                raise APIError(f'Paid request ceiling reached ({count}/{limit}); no new request sent. Progress saved.')
            return db.execute('INSERT INTO requests (started_at,model,status) VALUES (?,?,?)',
                              (time.time(),model,'in_flight')).lastrowid

    def finish(self, request_id, status, response=None):
        response=response or {}
        cost=(response.get('usage') or {}).get('cost')
        if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0: cost=None
        with contextlib.closing(sqlite3.connect(self.path,timeout=30)) as db, db:
            db.execute('UPDATE requests SET status=?,response_id=?,reported_usd=? WHERE id=?',
                       (status,response.get('id'),cost,request_id))

def api_cost_summary(store):

    from .common import PROTOCOL
    seen=set(); groups={}
    for kind in ('generation','translation','translation_error','backtranslation','validation'):
        for row in store.all(kind):
            if row.get('protocol')!=PROTOCOL: continue
            raw=row.get('raw') or {}
            if not raw: continue
            response_id=raw.get('id') or raw.get('responseId') or (row.get('created_at'),row.get('model'))
            identity=(row.get('provider'),response_id)
            if identity in seen: continue
            seen.add(identity)
            role='generation' if kind=='generation' else ('audit' if kind=='validation' else 'translation/back-translation')
            key=(role,row.get('provider'),row.get('model'))
            group=groups.setdefault(key,{'role':role,'provider':key[1],'model':key[2],
                                         'responses':0,'responses_with_cost':0,'reported_usd':0.})
            group['responses']+=1
            cost=(row.get('usage') or {}).get('cost')
            if isinstance(cost,(int,float)) and math.isfinite(cost) and cost>=0:
                group['responses_with_cost']+=1; group['reported_usd']+=cost
    rows=list(groups.values())
    for row in rows:
        if not row['responses_with_cost']: row['reported_usd']=None
    costs=[row['reported_usd'] for row in rows if row['reported_usd'] is not None]
    return {'groups':rows,'reported_usd':sum(costs) if costs else None,
            'note':'Provider-reported USD for all current-protocol responses retained in this run, including rejected translations. Missing costs are unknown, not zero. Deleted/backup responses and any unreported failed-request charges are excluded; the provider account remains the billing record.'}

class Client:
    def __init__(self, provider, c):
        self.provider, self.c = provider, c
        if c.get('mode')=='dev' and provider not in ('gemini','groq'):
            raise APIError('Development mode uses Gemini and direct Groq only. Use --mode prod for paid OpenRouter.')
        if c.get('mode')=='prod' and provider!='openrouter':
            raise APIError('Production uses pinned OpenRouter deployments only.')
        names = {'gemini': 'GEMINI_API_KEY', 'groq': 'GROQ_API_KEY', 'openrouter': 'OPENROUTER_API_KEY'}
        if provider not in names: raise APIError('Unsupported provider: '+provider)
        name = names[provider]
        self.key = os.getenv(name, '')
        if not self.key or self.key.startswith('your_'): raise APIError('Set '+name+' in .env')
        self.limiter = RateLimiter(provider, self.key, c[provider+'_rpm'])
        self.catalog = {}
        self.pricing = {}
        self._serial = threading.Lock()
        self._stop_error = None
        ledger=c.get('_paid_request_ledger')
        self.budget=PaidRequestBudget(ledger,c.get('_max_paid_requests')) if provider=='openrouter' and ledger and (
            c.get('_max_paid_requests') is not None or Path(ledger).exists()) else None

    def request(self, url, payload=None, metrics=None):

        with self._serial if self.provider=='groq' else contextlib.nullcontext():
            if self._stop_error: raise self._stop_error
            try:
                return self._request(url,payload,metrics)
            except APIError as error:
                if self.c.get('mode')=='dev': self._stop_error=error
                raise

    def _request(self, url, payload=None, metrics=None):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json',
                   'User-Agent': 'multilingual-codebench/2.0', 'Cache-Control': 'no-cache'}
        headers.update({'x-goog-api-key': self.key} if self.provider == 'gemini' else {'Authorization': 'Bearer '+self.key})
        if self.provider=='openrouter': headers['X-OpenRouter-Cache']='false'
        failures = []
        retry = None
        paid=self.provider=='openrouter' and payload is not None
        attempts=min(PAID_MAX_ATTEMPTS,self.c['max_attempts']) if paid else self.c['max_attempts']
        for attempt in range(attempts):
            if self._stop_error: raise self._stop_error
            pace = time.monotonic()
            self.limiter.acquire()
            if self._stop_error: raise self._stop_error
            if metrics is not None:
                metrics['pacing_seconds'] += time.monotonic()-pace
                metrics['attempts'] += 1
                metrics['request_started_at'].append(time.time())
            ticket=self.budget.reserve(payload.get('model')) if paid and self.budget else None
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
                started = time.monotonic()
                try:
                    with urllib.request.urlopen(req, timeout=300) as response:
                        data = json.load(response)
                        if metrics is not None and self.provider=='openrouter':
                            metrics['response_cache_status']=response.headers.get('X-OpenRouter-Cache-Status')
                        if self.provider=='groq' and payload is not None:
                            limits={key:response.headers[key] for key in (
                                'x-ratelimit-limit-requests','x-ratelimit-remaining-requests','x-ratelimit-reset-requests',
                                'x-ratelimit-limit-tokens','x-ratelimit-remaining-tokens','x-ratelimit-reset-tokens')
                                if response.headers.get(key) is not None}
                            if metrics is not None: metrics['rate_limit_headers']=limits
                            remaining=limits.get('x-ratelimit-remaining-tokens')
                            used=(data.get('usage') or {}).get('total_tokens') or 1024
                            if remaining is not None and float(remaining)<max(1024,used):
                                self.limiter.defer(reset_seconds(limits.get('x-ratelimit-reset-tokens'))+.2)
                finally:
                    if metrics is not None: metrics['http_seconds'] += time.monotonic()-started
                if ticket is not None: self.budget.finish(ticket,'provider_error' if 'error' in data else 'completed',data)
                if 'error' in data: raise APIError('Provider returned an error: '+str(data['error']).replace(self.key, '[redacted]')[:1000])
                if metrics is not None: metrics['transport_failures'] = failures
                return data
            except urllib.error.HTTPError as error:
                if ticket is not None: self.budget.finish(ticket,'http_'+str(error.code))
                detail = error.read().decode(errors='replace').replace(self.key, '[redacted]')

                daily = any(x in detail.lower() for x in ('per day', 'per-day', 'per_day', 'perday', 'daily', 'quota exhausted'))
                failures.append({'at': time.time(), 'http_status': error.code, 'detail': detail[:2000]})
                if metrics is not None: metrics['transport_failures'] = failures
                retryable=(429,503) if paid else (408,429,500,502,503,504)
                too_large='request too large' in detail.lower()
                server_delay=retry_after(error.headers.get('Retry-After'))
                match = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', detail)
                if match: server_delay=max(server_delay,float(match[1])+1)
                retry = ({'provider':self.provider,'kind':'daily_quota' if daily else
                          ('rate_limit' if error.code==429 else 'service'),
                          'delay_seconds':server_delay,'http_status':error.code}
                         if not paid and error.code in retryable and not too_large else None)
                if daily or too_large or error.code not in retryable:
                    raise APIError(f'{self.provider} HTTP {error.code}: {detail[:1500]}. Progress saved; resume the same run later.',retry=retry) from None


                delay = max(server_delay, min(240, 30*2**attempt) if paid else min(60, 3*2**attempt))
                if delay > (PAID_MAX_RETRY_WAIT_SECONDS if paid else 300):
                    raise APIError(f'Provider requires {delay:.0f}s waiting; progress saved. Resume later.',retry=retry) from None
                if error.code in (429,503): self.limiter.defer(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, json.JSONDecodeError, UnicodeDecodeError) as error:
                failures.append({'at': time.time(), 'type': type(error).__name__,
                                 'detail':str(error).replace(self.key,'[redacted]')[:500]})
                if ticket is not None: self.budget.finish(ticket,'uncertain_'+type(error).__name__)
                if paid:
                    raise APIError('OpenRouter response was lost or unreadable ('+type(error).__name__+
                                   '); not retried because it may already be billed. Progress saved; inspect the request ledger before resuming.') from None
                retry={'provider':self.provider,'kind':'service','delay_seconds':0}
                delay = min(60, 3*2**attempt)
            if attempt+1 < attempts:
                print(f'{self.provider}: transport retry {attempt+2}/{attempts} after {delay:.1f}s', flush=True)
                time.sleep(delay)
        last=failures[-1] if failures else {}
        detail=str(last.get('detail') or last.get('type') or 'unknown transport error')[:500]
        raise APIError(self.provider+': transport retry budget exhausted; progress saved. Last error: '+detail,retry=retry)

    def preflight(self, models):
        for model in dict.fromkeys(models):
            if self.provider == 'gemini':
                data = self.request('https://generativelanguage.googleapis.com/v1beta/models/'+model)
                if 'generateContent' not in data.get('supportedGenerationMethods', []):
                    raise APIError('Model does not support generateContent: '+model)
                item = {'provider': 'gemini', 'model': model, 'backend': 'Google AI Studio',
                        'revision': data.get('version'), 'name': data.get('name'),
                        'context_length': data['inputTokenLimit'], 'max_output_tokens': data['outputTokenLimit'],
                        'provider_default_temperature':data.get('temperature'),
                        'supports_seed': False, 'immutable_revision_available': False}
            elif self.provider == 'groq':
                data = self.request('https://api.groq.com/openai/v1/models/'+model)
                if not data.get('active',True): raise APIError('Groq model is inactive: '+model)
                item = {'provider':'groq','model':model,'backend':'Groq','provider_name':'Groq',
                        'revision':None,'name':data.get('name',data['id']),
                        'context_length':data['context_window'],
                        'max_output_tokens':data['max_completion_tokens'],
                        'supports_seed':'seed' in data.get('supported_sampling_parameters',[]),
                        'supports_json_output':'json_mode' in data.get('supported_features',[]),
                        'immutable_revision_available':False}
            else:
                if model.endswith(':free'): raise APIError('Use a paid OpenRouter model identifier for the main deployment')
                data = self.request('https://openrouter.ai/api/v1/models/'+model+'/endpoints')['data']
                endpoints = data['endpoints']

                standard = [e for e in endpoints if not e['tag'].endswith(('/flex','/priority','/fast'))]
                required = set(sampling_settings(self.provider, model)) | {'max_tokens'}
                if model in (self.c.get('translation_model'),self.c.get('validator_model')):
                    required.add('response_format')
                eligible = [e for e in endpoints if required <= set(e.get('supported_parameters', []))
                            and (e.get('max_completion_tokens') or self.c['max_tokens']) >= self.c['max_tokens']]
                backend = {'google/gemma-4-26b-a4b-it': 'parasail/bf16',
                           'openai/gpt-oss-20b': 'coreweave/fp4'}.get(
                               model, self.c.get('openrouter_backends', {}).get(model))
                if backend:
                    if '/' not in backend and sum(e['tag'].split('/')[0]==backend for e in standard)>1:
                        raise APIError('The base provider slug matches several endpoints. Select an exact variant tag for '+model)
                    choices = [e for e in eligible if e.get('tag') == backend]
                else:

                    choices = [e for e in eligible if e in standard and ('/' in e['tag'] or
                               sum(x['tag'].split('/')[0] == e['tag'] for x in standard) == 1)]
                    choices.sort(key=lambda e: e['tag'])
                if not choices:
                    raise APIError('No fixed endpoint supports the settings for '+model+'. Available tags: '+', '.join(e['tag'] for e in endpoints))
                e = choices[0]
                self.pricing[model] = e.get('pricing',{})
                item = {'provider': 'openrouter', 'model': model, 'backend': e['tag'],
                        'provider_name': e['provider_name'], 'revision': e['name'],
                        'quantization': e.get('quantization'), 'context_length': e['context_length'],
                        'max_output_tokens': e.get('max_completion_tokens'),
                        'supports_seed': 'seed' in e.get('supported_parameters', []),
                        'supports_json_output': 'response_format' in e.get('supported_parameters', []),
                        'immutable_revision_available': False}
            if item.get('max_output_tokens') and self.c['max_tokens'] > item['max_output_tokens']:
                raise APIError('Configured output limit exceeds deployment limit for '+model)
            item['settings'] = {**sampling_settings(self.provider, model), 'max_tokens': self.c['max_tokens']}
            self.catalog[model] = item
            if self.provider=='openrouter':
                prices=self.pricing[model]
                print(f'{model}: fixed backend {item["backend"]}; '
                      f'${float(prices.get("prompt",0))*1e6:g} input / '
                      f'${float(prices.get("completion",0))*1e6:g} output per million tokens',flush=True)
        return self.catalog

    def generate(self, model, text, system='Return only the requested output.', seed=None, json_output=False):
        if model not in self.catalog: raise APIError('Run provider preflight before generation')
        deployment = self.catalog[model]


        if len((text+system).encode('utf-8'))+512+self.c['max_tokens'] > deployment['context_length']:
            raise APIError('Prompt may exceed the context window for '+model+'; no truncation was applied')
        start = time.time()
        metrics = {'http_seconds': 0., 'pacing_seconds': 0., 'attempts': 0, 'request_started_at': [],
                   'http_timeout_seconds':300}
        settings = dict(deployment['settings'])
        request_seed = seed if deployment['supports_seed'] else None
        if self.provider == 'gemini':
            cfg = {'maxOutputTokens': settings['max_tokens']}
            if 'temperature' in settings: cfg['temperature']=settings['temperature']
            if 'thinking_level' in settings: cfg['thinkingConfig']={'thinkingLevel':settings['thinking_level']}
            for key, target in [('top_p','topP'), ('top_k','topK')]:
                if key in settings: cfg[target] = settings[key]
            if json_output: cfg['responseMimeType'] = 'application/json'
            payload = {'systemInstruction': {'parts': [{'text': system}]},
                       'contents': [{'role': 'user', 'parts': [{'text': text}]}], 'generationConfig': cfg}
            data = self.request('https://generativelanguage.googleapis.com/v1beta/models/'+model+':generateContent', payload, metrics)
            candidates = data.get('candidates') or []
            candidate = candidates[0] if candidates else {}
            output = ''.join(p.get('text') or '' for p in (candidate.get('content') or {}).get('parts') or [] if not p.get('thought'))
            finish, usage = candidate.get('finishReason'), data.get('usageMetadata') or {}
            revision = data.get('modelVersion')
            backend = deployment['backend']
        else:
            if json_output and not deployment.get('supports_json_output'):
                raise APIError('The fixed audit endpoint does not support JSON responses: '+model)
            payload = {'model': model, 'messages': [{'role':'system','content':system}, {'role':'user','content':text}],
                       **settings}
            if self.provider=='openrouter':
                route = deployment['backend']
                payload['provider']={'only':[route], 'order':[route], 'allow_fallbacks':False, 'require_parameters':True}
            if request_seed is not None: payload['seed'] = request_seed
            if json_output: payload['response_format'] = {'type':'json_object'}
            url = ('https://api.groq.com/openai/v1/chat/completions' if self.provider=='groq'
                   else 'https://openrouter.ai/api/v1/chat/completions')
            data = self.request(url, payload, metrics)
            choices = data.get('choices') or []
            candidate = choices[0] if choices else {}
            output = (candidate.get('message') or {}).get('content') or ''
            if not isinstance(output, str): output = ''
            finish, usage = candidate.get('finish_reason'), data.get('usage') or {}
            revision = data.get('model')
            backend = 'Groq' if self.provider=='groq' else data.get('provider')

        cached = any(data.get(k) is True for k in ('cached_response','response_cached','is_cached')) or (
            (metrics.get('response_cache_status') or '').upper()=='HIT')
        return {'text': output, 'raw': data, 'request': payload, 'model': model, 'provider': self.provider,
                'deployment': deployment, 'settings': settings, 'usage': usage, 'request_seed': request_seed,
                'quoted_pricing': self.pricing.get(model),
                'observed_revision': revision, 'observed_backend': backend, 'known_cached_response': cached,
                'prompt': text, 'system_prompt': system, 'prompt_source':'captured',
                'input_utf8_bytes': len(text.encode('utf-8')), 'input_characters': len(text),
                'seconds': time.time()-start, 'created_at': time.time(), 'finish_reason': finish,
                'request_metrics': metrics, 'no_text': not bool(output.strip())}
