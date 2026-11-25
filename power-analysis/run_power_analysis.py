from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t


SEED = 20261006
ALPHA = 0.05
LANGUAGES = 9
MODELS = 2
REPEATS = 3
BENCHMARKS = 3
BATCH = 250
CORE_REPLICATIONS = 5000
STRESS_REPLICATIONS = 2000


COUPLING = {
    "weaker_pairing": (0.10, 0.35, 0.00, 0.55),
    "stronger_pairing": (0.50, 0.25, 0.25, 0.00),
}
BENCHMARK_OFFSETS = np.array([-0.025, 0.0, 0.025])
MODEL_OFFSETS = np.array([-0.025, 0.025])
EFFECT_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)


@dataclass(frozen=True)
class Scenario:
    n_per_benchmark: int
    midpoint: float
    repeat_correlation: float
    coupling: str
    effect: float
    alternative: str = "single"
    missing_rate: float = 0.0
    missing_pattern: str = "uniform"
    heterogeneous_effect: bool = False
    replications: int = CORE_REPLICATIONS


def scenarios() -> list[Scenario]:
    specs = []
    for rho in (0.3, 0.6, 0.9, 1.0):
        for coupling in COUPLING:
            for midpoint in (0.2, 0.5, 0.8):
                common = dict(n_per_benchmark=50, midpoint=midpoint,
                              repeat_correlation=rho, coupling=coupling)
                specs.append(Scenario(**common, effect=0.0))
                specs.extend(Scenario(**common, effect=effect)
                             for effect in EFFECT_GRID)
                specs.append(Scenario(**common, effect=0.10, alternative="all"))
                specs.append(Scenario(**common, effect=-0.10))
                specs.append(Scenario(**common, effect=0.10,
                                      heterogeneous_effect=True))


                if rho in (0.6, 1.0):
                    for rate in (0.05, 0.10):
                        for pattern in ("uniform", "language_dependent"):
                            for effect in (0.0, 0.10):
                                specs.append(Scenario(
                                    **common, effect=effect, missing_rate=rate,
                                    missing_pattern=pattern,
                                    replications=STRESS_REPLICATIONS))


    for n in (100, 150):
        for rho in (0.3, 0.9, 1.0):
            for coupling in COUPLING:
                for effect in (0.0, 0.10):
                    specs.append(Scenario(n, 0.5, rho, coupling, effect))
    return specs


def generator_key(s: Scenario) -> tuple:
    return s.n_per_benchmark, s.repeat_correlation, s.coupling


def rng_for(key: tuple) -> np.random.Generator:
    digest = hashlib.sha256(repr((SEED, key)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def draw_uniforms(rng, batch: int, n: int, rho: float, coupling: str):
    shape = (batch, BENCHMARKS, n)
    component = rng.choice(4, size=shape, p=COUPLING[coupling])
    base = rng.random(shape + (LANGUAGES, MODELS))
    common = rng.random(shape + (1, 1))
    by_language = rng.random(shape + (LANGUAGES, 1))
    by_model = rng.random(shape + (1, MODELS))
    base = np.where((component == 0)[..., None, None], common, base)
    base = np.where((component == 1)[..., None, None], by_language, base)
    base = np.where((component == 2)[..., None, None], by_model, base)


    fresh = rng.random(shape + (LANGUAGES, MODELS, REPEATS))
    copies = rng.random(fresh.shape) < math.sqrt(rho)
    outcomes_uniform = np.where(copies, base[..., None], fresh)
    missing_uniform = rng.random(fresh.shape)
    return outcomes_uniform, missing_uniform


def probabilities(s: Scenario):
    midpoint = (s.midpoint + BENCHMARK_OFFSETS[:, None]
                + MODEL_OFFSETS[None, :])
    cell_effect = np.full((BENCHMARKS, MODELS), s.effect)
    if s.heterogeneous_effect:
        cell_effect *= np.array([0.8, 1.0, 1.2])[:, None]
        cell_effect *= np.array([0.75, 1.25])[None, :]
    english = midpoint - cell_effect / 2
    p = np.broadcast_to(english[:, None, :],
                        (BENCHMARKS, LANGUAGES, MODELS)).copy()
    if s.alternative == "single":
        p[:, 1, :] += cell_effect
    elif s.alternative == "all":
        p[:, 1:, :] += cell_effect[:, None, :]
    else:
        raise ValueError(s.alternative)
    if not ((p >= 0).all() and (p <= 1).all()):
        raise ValueError(f"Infeasible probabilities: {s}")
    truth = (p[:, 1:, :] - p[:, 0:1, :]).mean(axis=(0, 2))
    return p[None, :, None, :, :, None], truth


def complete_conditions(missing_uniform, s: Scenario):
    if s.missing_rate == 0:
        return np.ones(missing_uniform.shape[:-2], dtype=bool)
    rates = np.full(LANGUAGES, s.missing_rate)
    if s.missing_pattern == "language_dependent":
        rates[0] *= 0.5
        rates[1:] *= 1.5
    return (missing_uniform >= rates[None, None, None, :, None, None]).all(
        axis=(-1, -2))


def planned_test(counts, complete):


    counts = counts.astype(np.int16)
    d = (counts[..., 1:] - counts[..., 0:1]) / (MODELS * REPEATS)
    paired = complete[..., 1:] & complete[..., 0:1]
    n = paired.sum(axis=2)
    total = np.where(paired, d, 0.0).sum(axis=2)
    squared = np.where(paired, d * d, 0.0).sum(axis=2)
    safe_n = np.maximum(n, 2)
    means = total / safe_n
    variance = np.maximum((squared - total * total / safe_n)
                          / (safe_n - 1), 0.0)
    v = variance / (BENCHMARKS ** 2 * safe_n)
    estimate = means.mean(axis=1)
    se2 = v.sum(axis=1)
    denominator = (v * v / (safe_n - 1)).sum(axis=1)
    testable = (n >= 2).all(axis=1) & (se2 > 1e-15) & (denominator > 0)
    df = np.divide(se2 * se2, denominator,
                   out=np.ones_like(se2), where=denominator > 0)
    se = np.sqrt(se2)
    statistic = np.divide(estimate, se, out=np.zeros_like(se), where=se > 0)
    p = np.where(testable, 2 * t.sf(np.abs(statistic), df), 1.0)
    order = np.argsort(p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    thresholds = ALPHA / np.arange(LANGUAGES - 1, 0, -1)
    sorted_reject = np.logical_and.accumulate(sorted_p <= thresholds, axis=1)
    reject = np.zeros_like(sorted_reject)
    np.put_along_axis(reject, order, sorted_reject, axis=1)
    halfwidth = t.ppf(0.975, df) * se
    return estimate, halfwidth, testable, reject, n


def validate_generator():
    rho = 0.6
    coupling = "stronger_pairing"
    u, _ = draw_uniforms(rng_for(("validation",)), 1, 20000, rho, coupling)
    y = (u < 0.5).reshape(-1, LANGUAGES, MODELS, REPEATS)
    empirical = {
        "mean_success": float(y.mean()),
        "repeat_correlation": float(np.corrcoef(y[:, 0, 0, 0], y[:, 0, 0, 1])[0, 1]),
        "paired_language_correlation": float(np.corrcoef(y[:, 0, 0, 0], y[:, 1, 0, 0])[0, 1]),
        "task_difference_variance": float((y[:, 1].mean(axis=(1, 2))
                                         - y[:, 0].mean(axis=(1, 2))).var(ddof=1)),
    }
    a, b, c, _ = COUPLING[coupling]
    expected = {
        "mean_success": 0.5,
        "repeat_correlation": rho,
        "paired_language_correlation": rho * (a + c),
        "task_difference_variance": 0.25 * ((1 + 2 * rho) / 3 + rho * (b - c - a)),
    }
    for key, value in empirical.items():
        tolerance = 0.025 if "correlation" in key else 0.005
        if abs(value - expected[key]) > tolerance:
            raise RuntimeError(f"Generator check failed: {key}")
    return {"empirical": empirical, "expected": expected,
            "independent_validation_tasks": len(y)}


def run():
    started = time.perf_counter()
    specs = scenarios()
    validation = validate_generator()
    groups = defaultdict(list)
    for index, s in enumerate(specs):
        groups[generator_key(s)].append(index)
    accumulators = [defaultdict(float) for _ in specs]
    for group_number, (key, indices) in enumerate(groups.items(), 1):
        n, rho, coupling = key
        rng = rng_for(key)
        max_reps = max(specs[i].replications for i in indices)
        for start in range(0, max_reps, BATCH):
            batch = min(BATCH, max_reps - start)
            u, mu = draw_uniforms(rng, batch, n, rho, coupling)
            for i in indices:
                s = specs[i]
                if start >= s.replications:
                    continue
                take = min(batch, s.replications - start)
                p, truth = probabilities(s)
                counts = (u[:take] < p).sum(axis=(-1, -2), dtype=np.int8)
                complete = complete_conditions(mu[:take], s)
                estimate, halfwidth, testable, reject, pairs = planned_test(counts, complete)
                nulls = np.abs(truth) < 1e-12
                a = accumulators[i]
                a["replications"] += take
                a["target_rejections"] += reject[:, 0].sum()
                if nulls.any():
                    a["false_family_rejections"] += reject[:, nulls].any(axis=1).sum()
                a["true_nulls"] = nulls.sum()
                a["target_testable"] += testable[:, 0].sum()
                covered = (np.abs(estimate[:, 0] - truth[0]) <= halfwidth[:, 0])
                a["target_covered"] += (covered & testable[:, 0]).sum()
                a["target_halfwidth_sum"] += halfwidth[testable[:, 0], 0].sum()
                a["target_pair_count_sum"] += pairs[:, :, 0].sum()
                a["target_estimate_sum"] += estimate[testable[:, 0], 0].sum()
        elapsed = time.perf_counter() - started
        print(f"Completed generator group {group_number}/{len(groups)}: "
              f"N={BENCHMARKS*n}, rho={rho}, {coupling}; {elapsed:.1f}s", flush=True)

    rows = []
    for s, a in zip(specs, accumulators, strict=True):
        n_sim = int(a["replications"])
        rate = a["target_rejections"] / n_sim
        valid = int(a["target_testable"])
        fwer = a["false_family_rejections"] / n_sim if a["true_nulls"] else None
        coverage = a["target_covered"] / valid if valid else None
        rows.append({
            **asdict(s), "total_tasks": s.n_per_benchmark * BENCHMARKS,
            "target_rejection_rate": rate,
            "target_rejection_mcse": math.sqrt(rate * (1 - rate) / n_sim),
            "true_nulls": int(a["true_nulls"]), "family_wise_error_rate": fwer,
            "family_wise_error_mcse": (math.sqrt(fwer * (1 - fwer) / n_sim)
                                       if fwer is not None else None),
            "target_testable_rate": valid / n_sim,
            "pointwise_coverage_given_testable": coverage,
            "coverage_mcse": (math.sqrt(coverage * (1 - coverage) / valid)
                              if coverage is not None else None),
            "mean_pointwise_halfwidth": a["target_halfwidth_sum"] / valid if valid else None,
            "mean_available_paired_tasks": a["target_pair_count_sum"] / n_sim,
            "mean_target_estimate_given_testable": a["target_estimate_sum"] / valid if valid else None,
        })
    output = Path(__file__).resolve().parent
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "planning_date": "2026-10-06", "seed": SEED,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "scenarios": len(rows),
        "simulated_experiments": sum(s.replications for s in specs),
        "elapsed_seconds": time.perf_counter() - started,
        "coupling_weights": COUPLING, "generator_validation": validation,
        "definition": "Target power is the Holm-adjusted rejection rate for language 1, not probability of any significant comparison.",
        "no_actual_model_outputs_used": True,
        "common_random_numbers": "Within each generator group, probability/effect/missingness scenarios share pseudorandom uniforms. Each scenario has independent simulated experiments; scenarios are not independent of one another.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Saved {len(rows)} scenarios and metadata to {output}", flush=True)


if __name__ == "__main__":
    run()
