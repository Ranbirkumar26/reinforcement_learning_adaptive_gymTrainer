from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.coaching import write_policy_metadata
from src.rl_env import SyntheticSquatCoachEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SEEDS = (13, 17, 23, 29, 31)
DEFAULT_EVAL_SEEDS = tuple(range(1001, 1031))
DEFAULT_TIMESTEPS = 50_000
MONITOR_KEYS = ("final_fatigue", "final_risk", "final_repeated_mistakes", "correction_rate")


@dataclass(frozen=True)
class TrainingResult:
    canonical_policy: Path
    trained_models: list[Path]
    episode_rewards: Path
    manifest: Path


class EpisodeRewardLogger:
    def __init__(self, seed: int, rows: list[dict[str, float | int]]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-not-found]

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.episode_index = 0

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    episode = info.get("episode")
                    if episode is None:
                        continue
                    self.episode_index += 1
                    rows.append(
                        {
                            "seed": seed,
                            "episode": self.episode_index,
                            "timestep": int(self.num_timesteps),
                            "reward": round(float(episode["r"]), 6),
                            "length": int(episode["l"]),
                            "final_fatigue": round(float(episode.get("final_fatigue", 0.0)), 6),
                            "final_risk": round(float(episode.get("final_risk", 0.0)), 6),
                            "final_repeated_mistakes": round(
                                float(episode.get("final_repeated_mistakes", 0.0)),
                                6,
                            ),
                            "correction_rate": round(float(episode.get("correction_rate", 0.0)), 6),
                        }
                    )
                return True

        self.callback = _Callback()


def _library_versions() -> dict[str, str]:
    libraries = [
        "stable-baselines3",
        "gymnasium",
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "opencv-python",
        "mediapipe",
        "streamlit",
    ]
    versions: dict[str, str] = {}
    for package in libraries:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "episode",
        "timestep",
        "reward",
        "length",
        "final_fatigue",
        "final_risk",
        "final_repeated_mistakes",
        "correction_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    *,
    command: str,
    timesteps: int,
    train_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    trained_models: list[Path],
    status: str,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": status,
        "command": command,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": _library_versions(),
        "timesteps_per_seed": timesteps,
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "models": [_display_path(path) for path in trained_models],
        "reward_source": "Stable-Baselines3 Monitor episode rewards from simulator training.",
        "reward_config": "full_reward",
        "error": error,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train(
    output_path: Path = ROOT / "models" / "coach_policy",
    timesteps: int = DEFAULT_TIMESTEPS,
    seeds: Iterable[int] = DEFAULT_TRAIN_SEEDS,
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    output_dir: Path = ROOT / "outputs" / "rl_training",
    model_dir: Path = ROOT / "models" / "rl_runs",
    command: str | None = None,
) -> TrainingResult:
    seed_list = [int(seed) for seed in seeds]
    eval_seed_list = [int(seed) for seed in eval_seeds]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    episode_rewards_path = output_dir / "episode_rewards.csv"
    manifest_path = output_dir / "run_manifest.json"
    command = command or shlex.join([sys.executable, *sys.argv])
    episode_rows: list[dict[str, float | int]] = []
    trained_models: list[Path] = []

    try:
        from stable_baselines3 import DQN  # type: ignore[import-not-found]
        from stable_baselines3.common.monitor import Monitor  # type: ignore[import-not-found]
    except Exception as exc:
        write_policy_metadata(output_path)
        _write_rows(episode_rewards_path, [])
        _write_manifest(
            manifest_path,
            command=command,
            timesteps=0,
            train_seeds=seed_list,
            eval_seeds=eval_seed_list,
            trained_models=[],
            status="fallback_policy_written",
            error=str(exc),
        )
        return TrainingResult(output_path.with_suffix(".json"), [], episode_rewards_path, manifest_path)

    completed_timesteps = timesteps
    for index, seed in enumerate(seed_list):
        env = Monitor(
            SyntheticSquatCoachEnv(seed=seed),
            filename=None,
            info_keywords=MONITOR_KEYS,
        )
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            buffer_size=5000,
            learning_starts=min(100, max(1, timesteps // 10)),
            batch_size=32,
            gamma=0.92,
            verbose=0,
            seed=seed,
        )
        callback = EpisodeRewardLogger(seed, episode_rows).callback
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
        seed_model_base = model_dir / f"coach_policy_seed_{seed}"
        model.save(str(seed_model_base))
        seed_model_zip = seed_model_base.with_suffix(".zip")
        trained_models.append(seed_model_zip)
        if index == 0:
            model.save(str(output_path))

    _write_rows(episode_rewards_path, episode_rows)
    _write_manifest(
        manifest_path,
        command=command,
        timesteps=completed_timesteps,
        train_seeds=seed_list,
        eval_seeds=eval_seed_list,
        trained_models=trained_models,
        status="completed",
    )
    return TrainingResult(output_path.with_suffix(".zip"), trained_models, episode_rewards_path, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Adaptive RL Gym Coach policy.")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "coach_policy")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "rl_training")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "rl_runs")
    args = parser.parse_args()

    result = train(
        output_path=args.output,
        timesteps=args.timesteps,
        seeds=args.seeds,
        eval_seeds=args.eval_seeds,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    print(f"Saved canonical policy: {result.canonical_policy}")
    print(f"Saved {len(result.trained_models)} seed policies under: {args.model_dir}")
    print(f"Episode reward log: {result.episode_rewards}")
    print(f"Run manifest: {result.manifest}")


if __name__ == "__main__":
    main()
