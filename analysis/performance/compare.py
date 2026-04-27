from collections import defaultdict
from statistics import mean
from typing import Iterable, List, Sequence


def compute_baseline_deltas(
    generation_rows: Iterable[dict], baselines: Sequence[str]
) -> List[dict]:
    baseline_set = set(baselines)
    by_generation = defaultdict(list)
    for row in generation_rows:
        by_generation[int(row["generation"])].append(row)

    deltas = []
    for generation, rows in sorted(by_generation.items()):
        baseline_rows = [r for r in rows if r["strategy"] in baseline_set]
        if not baseline_rows:
            continue

        baseline_success_mean = mean([float(r["success_rate_percent_mean"]) for r in baseline_rows])
        baseline_eff_step_mean = mean([float(r["efficiency_success_per_step_mean"]) for r in baseline_rows])
        baseline_eff_sec_mean = mean([float(r["efficiency_success_per_second_mean"]) for r in baseline_rows])
        baseline_resource_mean = mean([float(r["resource_efficiency_mean"]) for r in baseline_rows])

        for row in rows:
            if row["strategy"] in baseline_set:
                continue
            success = float(row["success_rate_percent_mean"])
            eff_step = float(row["efficiency_success_per_step_mean"])
            eff_sec = float(row["efficiency_success_per_second_mean"])
            resource = float(row["resource_efficiency_mean"])

            deltas.append(
                {
                    "strategy": row["strategy"],
                    "generation": generation,
                    "baseline_success_rate_mean": baseline_success_mean,
                    "strategy_success_rate_mean": success,
                    "delta_success_rate": success - baseline_success_mean,
                    "relative_lift_success_rate_percent": _safe_relative_lift(success, baseline_success_mean),
                    "baseline_efficiency_per_step_mean": baseline_eff_step_mean,
                    "strategy_efficiency_per_step_mean": eff_step,
                    "delta_efficiency_per_step": eff_step - baseline_eff_step_mean,
                    "baseline_efficiency_per_second_mean": baseline_eff_sec_mean,
                    "strategy_efficiency_per_second_mean": eff_sec,
                    "delta_efficiency_per_second": eff_sec - baseline_eff_sec_mean,
                    "baseline_resource_efficiency_mean": baseline_resource_mean,
                    "strategy_resource_efficiency_mean": resource,
                    "delta_resource_efficiency": resource - baseline_resource_mean,
                }
            )
    return deltas


def rank_strategies(strategy_trial_summary: Iterable[dict]) -> List[dict]:
    rows = list(strategy_trial_summary)
    ranked = sorted(rows, key=lambda r: (float(r["auc_mean"]), float(r["final_success_mean"])), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _safe_relative_lift(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0
