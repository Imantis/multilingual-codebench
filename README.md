# Code generation experiment

The pipeline translates programming prompts, checks the translations, generates Python programs and evaluates them with benchmark tests and static measurements.

- `run.py` starts the pipeline or replays saved data.
- `project/` contains the pipeline, API clients, evaluator and analysis code; `scripts/` contains input-preparation helpers.
- `config.json` defines languages, models, backends and execution limits.
- `inputs.json.gz` contains the initial input selection and source hashes. Required benchmark inputs are restored under `data/` when starting a collection.
- `records.tar.gz` contains saved programs, prompt-preparation records, measurements and summaries.
- `power-analysis/` contains the sample-size simulation and its saved calculations; `licenses/` contains source attribution and redistribution notes.

## New collection

Use Python 3.12 or 3.13 and Docker with Linux containers. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

```sh
python -m pip install -r requirements.txt pyarrow==20.0.0
python run.py --mode prod --workdir runs/new --max-paid-requests 20000 --evaluation-workers 2
```

This uses paid OpenRouter endpoints. The request ceiling is 20,000 requests, not a dollar limit. Data are saved in `runs/new/experiment.sqlite`, with readable exports and a report under `runs/new/reports/`. Rerun the same command to resume.

## Saved data

Recompute saved comparisons without API keys or Docker:

```sh
python run.py --saved
```

The command extracts the archive into `reproduced/`, verifies the recalculated comparisons and writes plots and tables under `reproduced/figures/`.

Key files in the archive and report exports:

| File | Contents |
| --- | --- |
| `generated-code.jsonl` | Generated programs and model-response records. |
| `prepared-prompts.json` | Source prompts, accepted translations and check summaries. |
| `preparation-attempts.jsonl` | Translation-check and repair history. |
| `results.csv` | Per-program test outcomes and static measurements. |
| `selected-tasks.csv` | Task identifiers and source hashes. |
| `report.json`, `statistics.json` | Summary results and statistical comparisons. |

Some LiveCodeBench prompt text is represented by hashes; see `licenses/NOTICE.md`. Keep API keys in the ignored `.env` file.
