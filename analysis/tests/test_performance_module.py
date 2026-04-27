import json
import os
import tempfile
import unittest

from analysis.performance.aggregate import (
    aggregate_per_generation,
    aggregate_per_trial,
    rows_with_derived_metrics,
)
from analysis.performance.compare import compute_baseline_deltas
from analysis.performance.loaders import load_runs


class TestPerformanceModule(unittest.TestCase):
    def test_normalization_and_deltas_with_missing_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_random = self._make_baseline_run(tmp, "run_random", "random")
            run_spiking = self._make_nn_run(tmp, "run_spiking", "spiking")

            rows = load_runs([run_random, run_spiking])
            self.assertEqual(3, len(rows))
            self.assertEqual({"random", "spiking"}, {r.strategy for r in rows})

            enriched = rows_with_derived_metrics(rows)
            self.assertTrue(all("efficiency_success_per_step" in row for row in enriched))

            gen_summary = aggregate_per_generation(enriched)
            trial_summary = aggregate_per_trial(enriched, success_threshold=25.0)

            self.assertTrue(any(r["strategy"] == "random" and r["generation"] == 1 for r in gen_summary))
            self.assertTrue(any(r["strategy"] == "spiking" and r["generation"] == 1 for r in gen_summary))
            self.assertTrue(any(r["strategy"] == "spiking" and r["generation"] == 2 for r in gen_summary))

            deltas = compute_baseline_deltas(gen_summary, baselines=["random"])
            self.assertEqual(1, len(deltas), "Only generation 1 should have baseline+nonbaseline overlap")
            self.assertEqual(1, deltas[0]["generation"])
            self.assertAlmostEqual(15.0, deltas[0]["delta_success_rate"], places=6)

            self.assertEqual(2, len(trial_summary))
            self.assertTrue(any(row["strategy"] == "spiking" for row in trial_summary))

    def _make_baseline_run(self, root: str, run_name: str, strategy: str) -> str:
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        self._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "run_datetime": "2026-01-01T00:00:00",
                "strategy_name": strategy,
                "num_envs": 10,
                "max_steps": 1000,
                "continue_after_goal": False,
                "environment_parameters": {"map_image": "6.png"},
            },
        )

        trial_dir = os.path.join(run_dir, f"{strategy}_trial_1")
        gen1 = os.path.join(trial_dir, "gen_1")
        os.makedirs(gen1, exist_ok=True)
        # Intentionally omit generation 2 so delta computation can verify missing-generation handling.
        self._write_json(os.path.join(trial_dir, "trial_metadata.json"), {"max_generations": 2, "total_trials": 1})
        self._write_json(
            os.path.join(gen1, "log.json"),
            {
                "generation": 1,
                "trial": 1,
                "population_size": 10,
                "num_successful": 2,
                "success_rate_percent": 20.0,
                "avg_path_length": 50.0,
                "avg_energy_remaining": 10.0,
                "avg_health_remaining": 20.0,
                "avg_initial_distance_to_reward": 100.0,
                "generation_duration_seconds": 5.0,
                "total_steps_in_gen": 100,
                "extinct": False,
            },
        )
        return run_dir

    def _make_nn_run(self, root: str, run_name: str, strategy: str) -> str:
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        self._write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "run_datetime": "2026-01-01T00:00:00",
                "strategy_name": strategy,
                "num_envs": 10,
                "max_steps": 1000,
                "continue_after_goal": False,
                "environment_parameters": {"map_image": "6.png"},
            },
        )

        gen1 = os.path.join(run_dir, "trial_1", "gen_1")
        gen2 = os.path.join(run_dir, "trial_1", "gen_2")
        os.makedirs(gen1, exist_ok=True)
        os.makedirs(gen2, exist_ok=True)
        self._write_json(
            os.path.join(gen1, "log.json"),
            {
                "generation": 1,
                "trial": 1,
                "population_size": 10,
                "num_successful": 3,
                "success_rate_percent": 35.0,
                "avg_path_length": 45.0,
                "avg_energy_remaining": 12.0,
                "avg_health_remaining": 21.0,
                "avg_initial_distance_to_reward": 90.0,
                "generation_duration_seconds": 4.0,
                "total_steps_in_gen": 90,
                "curriculum_streak": 1,
                "curriculum_promoted": False,
                "goal_spawn_distance": 25.0,
            },
        )
        self._write_json(
            os.path.join(gen2, "log.json"),
            {
                "generation": 2,
                "trial": 1,
                "population_size": 10,
                "num_successful": 4,
                "success_rate_percent": 40.0,
                "avg_path_length": 42.0,
                "avg_energy_remaining": 13.0,
                "avg_health_remaining": 22.0,
                "avg_initial_distance_to_reward": 88.0,
                "generation_duration_seconds": 3.8,
                "total_steps_in_gen": 85,
                "curriculum_streak": 2,
                "curriculum_promoted": True,
                "goal_spawn_distance": 30.0,
            },
        )
        return run_dir

    def _write_json(self, path: str, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)


if __name__ == "__main__":
    unittest.main()
