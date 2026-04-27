import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from .schema import GenerationMetricRow


def rows_with_derived_metrics(rows: Iterable[GenerationMetricRow]) -> List[dict]:
    enriched = []
    for row in rows:
        item = row.to_dict()
        steps = max(int(item["total_steps_in_gen"]), 1)
        seconds = max(float(item["generation_duration_seconds"]), 1e-9)
        path_len = max(float(item["avg_path_length"]), 1.0)

        item["efficiency_success_per_step"] = float(item["num_successful"]) / steps
        item["efficiency_success_per_second"] = float(item["num_successful"]) / seconds
        item["resource_efficiency"] = float(item["success_rate_percent"]) / path_len
        enriched.append(item)
    return enriched


def aggregate_per_generation(rows: Iterable[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["generation"])].append(row)

    metrics = [
        "success_rate_percent",
        "num_successful",
        "avg_path_length",
        "avg_energy_remaining",
        "avg_health_remaining",
        "generation_duration_seconds",
        "total_steps_in_gen",
        "efficiency_success_per_step",
        "efficiency_success_per_second",
        "resource_efficiency",
    ]

    out = []
    for (strategy, generation), items in sorted(groups.items()):
        entry = {
            "strategy": strategy,
            "generation": generation,
            "n_trials_contributing": len(items),
        }
        for metric in metrics:
            vals = [float(i[metric]) for i in items]
            metric_mean = mean(vals)
            metric_std = pstdev(vals) if len(vals) > 1 else 0.0
            entry[f"{metric}_mean"] = metric_mean
            entry[f"{metric}_std"] = metric_std
        success_mean = entry["success_rate_percent_mean"]
        success_std = entry["success_rate_percent_std"]
        entry["stability_cv_success"] = (success_std / success_mean) if success_mean > 0 else 0.0
        out.append(entry)
    return out


def aggregate_per_trial(rows: Iterable[dict], success_threshold: float = 25.0) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["trial"])].append(row)

    out = []
    for (strategy, trial), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: int(r["generation"]))
        success = [float(r["success_rate_percent"]) for r in items]
        auc = _auc_over_generations(success)
        hit_threshold = [int(r["generation"]) for r in items if float(r["success_rate_percent"]) >= success_threshold]
        out.append(
            {
                "strategy": strategy,
                "trial": trial,
                "generations_observed": len(items),
                "area_under_learning_curve": auc,
                "time_to_threshold_gen": min(hit_threshold) if hit_threshold else None,
                "mean_success_rate_percent": mean(success) if success else 0.0,
                "final_success_rate_percent": success[-1] if success else 0.0,
                "mean_efficiency_success_per_step": mean(
                    [float(r["efficiency_success_per_step"]) for r in items]
                )
                if items
                else 0.0,
                "mean_efficiency_success_per_second": mean(
                    [float(r["efficiency_success_per_second"]) for r in items]
                )
                if items
                else 0.0,
            }
        )
    return out


def summarize_trials_by_strategy(trial_rows: Iterable[dict]) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in trial_rows:
        groups[row["strategy"]].append(row)

    out = []
    for strategy, items in sorted(groups.items()):
        aucs = [float(i["area_under_learning_curve"]) for i in items]
        finals = [float(i["final_success_rate_percent"]) for i in items]
        thresholds = [
            float(i["time_to_threshold_gen"])
            for i in items
            if i["time_to_threshold_gen"] is not None and not math.isnan(float(i["time_to_threshold_gen"]))
        ]
        out.append(
            {
                "strategy": strategy,
                "num_trials": len(items),
                "auc_mean": mean(aucs) if aucs else 0.0,
                "auc_std": pstdev(aucs) if len(aucs) > 1 else 0.0,
                "final_success_mean": mean(finals) if finals else 0.0,
                "time_to_threshold_mean": mean(thresholds) if thresholds else None,
            }
        )
    return out


def _auc_over_generations(success_values: List[float]) -> float:
    if not success_values:
        return 0.0
    if len(success_values) == 1:
        return success_values[0]
    auc = 0.0
    for i in range(1, len(success_values)):
        auc += (success_values[i - 1] + success_values[i]) * 0.5
    return auc
