import glob
import json
import os
from typing import Iterable, List

from .schema import GenerationMetricRow


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _meta_value(run_meta: dict, key: str):
    if key in run_meta:
        return run_meta[key]
    env_params = run_meta.get("environment_parameters", {})
    return env_params.get(key)


def discover_run_dirs(root_dir: str) -> List[str]:
    """Find run directories under root that contain metadata.json."""
    if not os.path.isdir(root_dir):
        return []

    discovered = []
    for entry in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, entry)
        if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "metadata.json")):
            discovered.append(run_dir)
    return sorted(discovered)


def load_runs(run_dirs: Iterable[str]) -> List[GenerationMetricRow]:
    rows: List[GenerationMetricRow] = []
    for run_dir in run_dirs:
        rows.extend(load_single_run(run_dir))
    return rows


def load_single_run(run_dir: str) -> List[GenerationMetricRow]:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing metadata.json in run directory: {run_dir}")

    run_meta = _read_json(metadata_path)
    strategy = run_meta.get("strategy_name", "")

    if strategy in {"random", "levy", "uniform"}:
        return _load_baseline_style_run(run_dir, run_meta)

    if strategy in {"spiking", "random_nn", "manual"}:
        return _load_nn_style_run(run_dir, run_meta)

    # Fallback: detect by trial folder naming pattern.
    if glob.glob(os.path.join(run_dir, "*_trial_*")):
        return _load_baseline_style_run(run_dir, run_meta)
    if glob.glob(os.path.join(run_dir, "trial_*")):
        return _load_nn_style_run(run_dir, run_meta)

    return []


def _load_baseline_style_run(run_dir: str, run_meta: dict) -> List[GenerationMetricRow]:
    rows: List[GenerationMetricRow] = []
    strategy = run_meta.get("strategy_name", "")
    run_id = os.path.basename(run_dir.rstrip("\\/"))

    trial_dirs = sorted(glob.glob(os.path.join(run_dir, f"{strategy}_trial_*")))
    for trial_dir in trial_dirs:
        trial_meta = {}
        trial_meta_path = os.path.join(trial_dir, "trial_metadata.json")
        if os.path.isfile(trial_meta_path):
            trial_meta = _read_json(trial_meta_path)

        for gen_dir in sorted(glob.glob(os.path.join(trial_dir, "gen_*"))):
            gen_log_path = os.path.join(gen_dir, "log.json")
            if not os.path.isfile(gen_log_path):
                continue

            gen = _read_json(gen_log_path)
            rows.append(
                GenerationMetricRow(
                    run_id=run_id,
                    run_dir=run_dir,
                    strategy=strategy,
                    trial=int(gen.get("trial", _parse_suffix_int(trial_dir))),
                    generation=int(gen.get("generation", _parse_suffix_int(gen_dir))),
                    population_size=int(gen.get("population_size", run_meta.get("num_envs", 0))),
                    num_successful=int(gen.get("num_successful", 0)),
                    success_rate_percent=float(gen.get("success_rate_percent", 0.0)),
                    avg_path_length=float(gen.get("avg_path_length", 0.0)),
                    avg_energy_remaining=float(gen.get("avg_energy_remaining", 0.0)),
                    avg_health_remaining=float(gen.get("avg_health_remaining", 0.0)),
                    avg_initial_distance_to_reward=float(gen.get("avg_initial_distance_to_reward", 0.0)),
                    generation_duration_seconds=float(gen.get("generation_duration_seconds", 0.0)),
                    total_steps_in_gen=int(gen.get("total_steps_in_gen", 0)),
                    extinct=bool(gen.get("extinct", False)),
                    run_datetime=run_meta.get("run_datetime", ""),
                    map_image=_meta_value(run_meta, "map_image") or "",
                    max_steps=_meta_value(run_meta, "max_steps"),
                    num_envs=_meta_value(run_meta, "num_envs"),
                    continue_after_goal=_meta_value(run_meta, "continue_after_goal"),
                    trial_max_generations=trial_meta.get("max_generations"),
                    trial_total_trials=trial_meta.get("total_trials"),
                )
            )
    return rows


def _load_nn_style_run(run_dir: str, run_meta: dict) -> List[GenerationMetricRow]:
    rows: List[GenerationMetricRow] = []
    strategy = run_meta.get("strategy_name", "")
    run_id = os.path.basename(run_dir.rstrip("\\/"))

    for trial_dir in sorted(glob.glob(os.path.join(run_dir, "trial_*"))):
        trial_num = _parse_suffix_int(trial_dir)
        for gen_dir in sorted(glob.glob(os.path.join(trial_dir, "gen_*"))):
            gen_log_path = os.path.join(gen_dir, "log.json")
            if not os.path.isfile(gen_log_path):
                continue
            gen = _read_json(gen_log_path)

            rows.append(
                GenerationMetricRow(
                    run_id=run_id,
                    run_dir=run_dir,
                    strategy=strategy,
                    trial=int(gen.get("trial", trial_num)),
                    generation=int(gen.get("generation", _parse_suffix_int(gen_dir))),
                    population_size=int(gen.get("population_size", run_meta.get("num_envs", 0))),
                    num_successful=int(gen.get("num_successful", 0)),
                    success_rate_percent=float(gen.get("success_rate_percent", 0.0)),
                    avg_path_length=float(gen.get("avg_path_length", 0.0)),
                    avg_energy_remaining=float(gen.get("avg_energy_remaining", 0.0)),
                    avg_health_remaining=float(gen.get("avg_health_remaining", 0.0)),
                    avg_initial_distance_to_reward=float(gen.get("avg_initial_distance_to_reward", 0.0)),
                    generation_duration_seconds=float(gen.get("generation_duration_seconds", 0.0)),
                    total_steps_in_gen=int(gen.get("total_steps_in_gen", 0)),
                    extinct=bool(gen.get("num_successful", 0) == 0),
                    goal_spawn_distance=(
                        float(gen["goal_spawn_distance"])
                        if gen.get("goal_spawn_distance") is not None
                        else None
                    ),
                    curriculum_streak=(
                        int(gen["curriculum_streak"])
                        if gen.get("curriculum_streak") is not None
                        else None
                    ),
                    curriculum_promoted=(
                        bool(gen["curriculum_promoted"])
                        if gen.get("curriculum_promoted") is not None
                        else None
                    ),
                    run_datetime=run_meta.get("run_datetime", ""),
                    map_image=_meta_value(run_meta, "map_image") or "",
                    max_steps=_meta_value(run_meta, "max_steps"),
                    num_envs=_meta_value(run_meta, "num_envs"),
                    continue_after_goal=_meta_value(run_meta, "continue_after_goal"),
                )
            )
    return rows


def _parse_suffix_int(path: str) -> int:
    stem = os.path.basename(path.rstrip("\\/"))
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0
