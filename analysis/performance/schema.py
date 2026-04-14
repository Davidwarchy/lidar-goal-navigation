from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class GenerationMetricRow:
    """Canonical per-generation row used by the performance analysis module."""

    run_id: str
    run_dir: str
    strategy: str
    trial: int
    generation: int
    population_size: int
    num_successful: int
    success_rate_percent: float
    avg_path_length: float
    avg_energy_remaining: float
    avg_health_remaining: float
    avg_initial_distance_to_reward: float
    generation_duration_seconds: float
    total_steps_in_gen: int
    extinct: bool = False
    goal_spawn_distance: Optional[float] = None
    curriculum_streak: Optional[int] = None
    curriculum_promoted: Optional[bool] = None

    run_datetime: str = ""
    map_image: str = ""
    max_steps: Optional[int] = None
    num_envs: Optional[int] = None
    continue_after_goal: Optional[bool] = None
    trial_max_generations: Optional[int] = None
    trial_total_trials: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def canonical_headers() -> list[str]:
    return list(GenerationMetricRow.__dataclass_fields__.keys())
