from __future__ import annotations

import csv
import json
from pathlib import Path

from src.actions import ACTION_NAMES
from src.evaluate import evaluate
from src.rl_env import REWARD_CONFIGS, SyntheticSquatCoachEnv, heuristic_action_from_state
from src.train_rl import train


def test_env_contract_stays_backward_compatible() -> None:
    env = SyntheticSquatCoachEnv(seed=13)
    obs, _ = env.reset(seed=13)

    assert tuple(obs.shape) == (9,)
    assert env.action_space.n == 8
    assert len(ACTION_NAMES) == 8


def test_seed_replay_produces_deterministic_trajectory() -> None:
    actions = [5, 0, 1, 4, 2]

    def rollout() -> list[tuple[list[float], float]]:
        env = SyntheticSquatCoachEnv(seed=23)
        obs, _ = env.reset(seed=1001)
        values: list[tuple[list[float], float]] = [(obs.tolist(), 0.0)]
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            values.append((obs.tolist(), reward))
            if terminated or truncated:
                break
        return values

    assert rollout() == rollout()


def test_reward_ablation_disables_only_named_component() -> None:
    seed = 1002
    full_env = SyntheticSquatCoachEnv(seed=seed, reward_config=REWARD_CONFIGS["full_reward"])
    no_fatigue_env = SyntheticSquatCoachEnv(seed=seed, reward_config=REWARD_CONFIGS["no_fatigue_penalty"])
    full_obs, _ = full_env.reset(seed=seed)
    no_fatigue_obs, _ = no_fatigue_env.reset(seed=seed)
    action = heuristic_action_from_state(full_obs.tolist())

    _, full_reward, _, _, full_info = full_env.step(action)
    _, no_fatigue_reward, _, _, no_fatigue_info = no_fatigue_env.step(action)

    assert full_obs.tolist() == no_fatigue_obs.tolist()
    assert full_info["reward_terms"]["fatigue_penalty"] < 0
    assert no_fatigue_info["reward_terms"]["fatigue_penalty"] == 0.0
    assert round(no_fatigue_reward - full_reward, 6) == round(
        -full_info["reward_terms"]["fatigue_penalty"],
        6,
    )


def test_train_and_evaluate_write_rl_evidence_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "rl_training"
    model_dir = tmp_path / "models"
    result = train(
        timesteps=96,
        seeds=(13,),
        eval_seeds=(1001, 1002),
        output_dir=output_dir,
        model_dir=model_dir,
        output_path=tmp_path / "coach_policy",
        command="python -m src.train_rl --timesteps 96 --seeds 13 --eval-seeds 1001 1002",
    )

    assert result.canonical_policy.exists()
    if result.trained_models:
        assert result.trained_models[0].exists()
    else:
        assert result.canonical_policy.suffix == ".json"
    assert result.episode_rewards.exists()

    paths = evaluate(
        tmp_path,
        model_dir=model_dir,
        train_seeds=(13,),
        eval_seeds=(1001, 1002),
        command="python -m src.evaluate --eval-seeds 1001 1002",
    )

    for key in (
        "per_seed_eval_metrics",
        "policy_comparison",
        "reward_ablation",
        "rl_reward_curve",
        "baseline_comparison",
        "action_distribution",
        "run_manifest",
    ):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0

    with paths["policy_comparison"].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "reward",
        "final_fatigue",
        "final_risk",
        "repeated_mistake_rate",
        "correction_rate",
    }
    assert all(row["winner"] in {"dqn", "heuristic", "random"} for row in rows)

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    assert manifest["train_seeds"] == [13]
    assert manifest["eval_seeds"] == [1001, 1002]
