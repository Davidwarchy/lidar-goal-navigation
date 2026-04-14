import csv
import json
import os
from typing import Iterable, List


def write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    headers = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_report_bundle(
    output_dir: str,
    per_generation_rows: List[dict],
    per_trial_rows: List[dict],
    baseline_delta_rows: List[dict],
    ranked_summary: List[dict],
    config: dict,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    generation_csv = os.path.join(output_dir, "per_generation_comparison.csv")
    trial_csv = os.path.join(output_dir, "per_trial_summary.csv")
    deltas_csv = os.path.join(output_dir, "baseline_deltas.csv")
    report_json = os.path.join(output_dir, "report.json")

    write_csv(generation_csv, per_generation_rows)
    write_csv(trial_csv, per_trial_rows)
    write_csv(deltas_csv, baseline_delta_rows)

    report_obj = {
        "config": config,
        "counts": {
            "per_generation_rows": len(per_generation_rows),
            "per_trial_rows": len(per_trial_rows),
            "baseline_delta_rows": len(baseline_delta_rows),
        },
        "strategy_ranking": ranked_summary,
    }
    write_json(report_json, report_obj)

    return {
        "per_generation_csv": generation_csv,
        "per_trial_csv": trial_csv,
        "baseline_deltas_csv": deltas_csv,
        "report_json": report_json,
    }


def maybe_write_plots(output_dir: str, per_generation_rows: List[dict], baseline_deltas: List[dict]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(output_dir, exist_ok=True)
    written = []

    # Success vs generation by strategy.
    by_strategy = {}
    for row in per_generation_rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy, items in sorted(by_strategy.items()):
        items = sorted(items, key=lambda r: int(r["generation"]))
        ax.plot(
            [int(r["generation"]) for r in items],
            [float(r["success_rate_percent_mean"]) for r in items],
            label=strategy,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Success Rate Mean (%)")
    ax.set_title("Success Rate vs Generation")
    ax.legend(loc="best", fontsize=8)
    p1 = os.path.join(output_dir, "success_vs_generation.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(p1)

    # Efficiency vs generation by strategy.
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy, items in sorted(by_strategy.items()):
        items = sorted(items, key=lambda r: int(r["generation"]))
        ax.plot(
            [int(r["generation"]) for r in items],
            [float(r["efficiency_success_per_step_mean"]) for r in items],
            label=strategy,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Successes per Step (Mean)")
    ax.set_title("Efficiency vs Generation")
    ax.legend(loc="best", fontsize=8)
    p2 = os.path.join(output_dir, "efficiency_vs_generation.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    written.append(p2)

    # Baseline deltas by strategy.
    by_strategy_delta = {}
    for row in baseline_deltas:
        by_strategy_delta.setdefault(row["strategy"], []).append(row)
    if by_strategy_delta:
        fig, ax = plt.subplots(figsize=(8, 4))
        for strategy, items in sorted(by_strategy_delta.items()):
            items = sorted(items, key=lambda r: int(r["generation"]))
            ax.plot(
                [int(r["generation"]) for r in items],
                [float(r["delta_success_rate"]) for r in items],
                label=strategy,
            )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Delta Success Rate (%)")
        ax.set_title("Success Delta vs Baseline")
        ax.legend(loc="best", fontsize=8)
        p3 = os.path.join(output_dir, "baseline_delta_vs_generation.png")
        fig.tight_layout()
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        written.append(p3)

    return written
