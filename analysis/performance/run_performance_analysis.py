import argparse
import os

from .aggregate import (
    aggregate_per_generation,
    aggregate_per_trial,
    rows_with_derived_metrics,
    summarize_trials_by_strategy,
)
from .compare import compute_baseline_deltas, rank_strategies
from .loaders import discover_run_dirs, load_runs
from .report import maybe_write_plots, write_report_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-strategy performance analyzer.")
    parser.add_argument(
        "--root",
        type=str,
        default="output",
        help="Root directory containing run subdirectories with metadata.json",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional explicit run directories. If set, these are used in addition to --root discovery.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="random,levy,uniform",
        help="Comma-separated baseline strategy names.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=25.0,
        help="Threshold for time_to_threshold_gen metric.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("analysis", "performance", "results"),
        help="Directory where CSV/JSON and optional plots are written.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate comparison plots if matplotlib is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovered = discover_run_dirs(args.root)
    run_dirs = sorted(set(discovered + list(args.runs)))

    if not run_dirs:
        raise SystemExit("No run directories found. Use --root and/or --runs.")

    rows = load_runs(run_dirs)
    if not rows:
        raise SystemExit("No generation rows were loaded from selected run directories.")

    enriched_rows = rows_with_derived_metrics(rows)
    per_generation = aggregate_per_generation(enriched_rows)
    per_trial = aggregate_per_trial(enriched_rows, success_threshold=args.success_threshold)
    strategy_summary = summarize_trials_by_strategy(per_trial)
    ranked = rank_strategies(strategy_summary)
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    deltas = compute_baseline_deltas(per_generation, baselines=baselines)

    output_paths = write_report_bundle(
        output_dir=args.output_dir,
        per_generation_rows=per_generation,
        per_trial_rows=per_trial,
        baseline_delta_rows=deltas,
        ranked_summary=ranked,
        config={
            "root": args.root,
            "runs": run_dirs,
            "baselines": baselines,
            "success_threshold": args.success_threshold,
        },
    )

    plot_paths = maybe_write_plots(args.output_dir, per_generation, deltas) if args.plots else []

    print("Performance analysis complete.")
    for key, val in output_paths.items():
        print(f"- {key}: {val}")
    for path in plot_paths:
        print(f"- plot: {path}")


if __name__ == "__main__":
    main()
