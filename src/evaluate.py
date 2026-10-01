from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Callable, Iterable

from src.actions import ACTION_NAMES, action_name
from src.analysis import analyze_landmark_csv, bundled_synthetic_landmarks
from src.coaching import HeuristicPolicy, make_coaching_outputs
from src.io_utils import write_coaching_outputs
from src.profiles import load_profile
from src.rl_env import (
    REWARD_CONFIGS,
    SyntheticSquatCoachEnv,
    heuristic_action_from_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SEEDS = (13, 17, 23, 29, 31)
DEFAULT_EVAL_SEEDS = tuple(range(1001, 1031))
METRIC_DIRECTIONS = {
    "reward": "max",
    "final_fatigue": "min",
    "final_risk": "min",
    "repeated_mistake_rate": "min",
    "correction_rate": "min",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _read_episode_rewards(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "seed": float(row["seed"]),
                "episode": float(row["episode"]),
                "timestep": float(row["timestep"]),
                "reward": float(row["reward"]),
            }
            for row in reader
        ]


def _write_svg_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    values: list[tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    pad = 56
    if not values:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{title}</text>
  <text x="{width/2}" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="18" fill="#6b7280">No logged training data available.</text>
</svg>
"""
        path.write_text(svg, encoding="utf-8")
        return

    min_x = min(x for x, _ in values)
    max_x = max(x for x, _ in values)
    min_y = min(y for _, y in values)
    max_y = max(y for _, y in values)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    def point(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - min_x) / span_x * (width - pad * 2)
        py = height - pad - (y - min_y) / span_y * (height - pad * 2)
        return px, py

    points = " ".join(f"{point(x, y)[0]:.1f},{point(x, y)[1]:.1f}" for x, y in values)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{title}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#172033" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#172033" stroke-width="1.5"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="4" points="{points}"/>
  <text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">{x_label}</text>
  <text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">{y_label}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    pad = 58
    max_value = max(values) if values else 1.0
    min_value = min(values) if values else 0.0
    baseline_value = min(0.0, min_value)
    upper_value = max(0.0, max_value)
    span = max(1e-6, upper_value - baseline_value)
    bar_width = (width - pad * 2) / max(1, len(labels)) * 0.58
    step = (width - pad * 2) / max(1, len(labels))
    plot_height = height - pad * 2

    def y_for(value: float) -> float:
        return pad + (upper_value - value) / span * plot_height

    zero_y = y_for(0.0)
    bars: list[str] = []
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        x = pad + index * step + (step - bar_width) / 2
        value_y = y_for(value)
        y = min(value_y, zero_y)
        bar_height = abs(zero_y - value_y)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">{label}</text>'
        )
        value_label_y = y - 8 if value >= 0 else min(height - 82, y + bar_height + 18)
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{max(24, value_label_y):.1f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#172033">{value:.3f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{title}</text>
  <line x1="{pad}" y1="{zero_y:.1f}" x2="{width-pad}" y2="{zero_y:.1f}" stroke="#172033" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#172033" stroke-width="1.5"/>
  {''.join(bars)}
  <text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">{y_label}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_action_distribution(path: Path, rows: list[dict[str, object]]) -> None:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts = json.loads(str(row["action_counts"]))
        for action, count in counts.items():
            totals[str(row["policy"])][action] += int(count)

    labels = list(ACTION_NAMES)
    policies = [policy for policy in ("dqn", "heuristic", "random") if policy in totals]
    width, height = 980, 480
    pad = 72
    colors = {"dqn": "#2563eb", "heuristic": "#059669", "random": "#dc2626"}
    max_count = max([totals[policy][action] for policy in policies for action in labels] or [1])
    group_width = (width - pad * 2) / max(1, len(labels))
    bar_width = group_width / max(1, len(policies) + 1)
    bars: list[str] = []
    for action_index, action in enumerate(labels):
        for policy_index, policy in enumerate(policies):
            value = totals[policy][action]
            x = pad + action_index * group_width + (policy_index + 0.25) * bar_width
            bar_height = value / max_count * (height - pad * 2)
            y = height - pad - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.82:.1f}" height="{bar_height:.1f}" fill="{colors[policy]}"/>'
            )
        bars.append(
            f'<text x="{pad + action_index * group_width + group_width / 2:.1f}" y="{height - 36}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151" transform="rotate(-25 {pad + action_index * group_width + group_width / 2:.1f} {height - 36})">{action}</text>'
        )
    legend = " ".join(
        f'<rect x="{pad + idx * 120}" y="50" width="14" height="14" fill="{colors[policy]}"/><text x="{pad + 20 + idx * 120}" y="62" font-family="Arial" font-size="13" fill="#374151">{policy}</text>'
        for idx, policy in enumerate(policies)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#172033">Action distribution on held-out simulator seeds</text>
  {legend}
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#172033" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#172033" stroke-width="1.5"/>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _run_episode(
    policy_name: str,
    policy_fn: Callable[[object, Random], int],
    eval_seed: int,
    *,
    train_seed: int | str,
    reward_config_name: str = "full_reward",
) -> dict[str, object]:
    env = SyntheticSquatCoachEnv(seed=eval_seed, reward_config=REWARD_CONFIGS[reward_config_name])
    obs, _ = env.reset(seed=eval_seed)
    rng = Random(eval_seed + 91)
    done = False
    total_reward = 0.0
    steps = 0
    action_counts: Counter[str] = Counter()
    last_info: dict[str, object] = {}
    while not done:
        action = policy_fn(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        action_counts[action_name(action)] += 1
        done = bool(terminated or truncated)
        last_info = info
    return {
        "policy": policy_name,
        "train_seed": train_seed,
        "reward_config": reward_config_name,
        "eval_seed": eval_seed,
        "episode_reward": round(total_reward, 6),
        "final_fatigue": round(float(last_info.get("final_fatigue", 0.0)), 6),
        "final_risk": round(float(last_info.get("final_risk", 0.0)), 6),
        "repeated_mistakes": round(float(last_info.get("final_repeated_mistakes", 0.0)), 6),
        "repeated_mistake_rate": round(float(last_info.get("final_repeated_mistakes", 0.0)) / max(1, steps), 6),
        "correction_rate": round(float(last_info.get("correction_rate", 0.0)), 6),
        "steps": steps,
        "action_counts": json.dumps(dict(action_counts), sort_keys=True),
    }


def _write_per_seed_metrics(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), str(row["train_seed"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (policy, train_seed), values in sorted(grouped.items()):
        rewards = [float(row["episode_reward"]) for row in values]
        summaries.append(
            {
                "policy": policy,
                "train_seed": train_seed,
                "eval_episodes": len(values),
                "mean_reward": round(_mean(rewards), 6),
                "std_reward": round(_std(rewards), 6),
                "mean_final_fatigue": round(_mean([float(row["final_fatigue"]) for row in values]), 6),
                "mean_final_risk": round(_mean([float(row["final_risk"]) for row in values]), 6),
                "mean_repeated_mistake_rate": round(
                    _mean([float(row["repeated_mistake_rate"]) for row in values]),
                    6,
                ),
                "mean_correction_rate": round(_mean([float(row["correction_rate"]) for row in values]), 6),
            }
        )
    _write_csv(path, summaries)
    return summaries


def _write_policy_comparison(
    path: Path,
    rows: list[dict[str, object]],
    per_seed_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    policies = [policy for policy in ("dqn", "heuristic", "random") if any(row["policy"] == policy for row in rows)]
    output_rows: list[dict[str, object]] = []
    metric_columns = {
        "reward": "episode_reward",
        "final_fatigue": "final_fatigue",
        "final_risk": "final_risk",
        "repeated_mistake_rate": "repeated_mistake_rate",
        "correction_rate": "correction_rate",
    }
    summary_columns = {
        "reward": "mean_reward",
        "final_fatigue": "mean_final_fatigue",
        "final_risk": "mean_final_risk",
        "repeated_mistake_rate": "mean_repeated_mistake_rate",
        "correction_rate": "mean_correction_rate",
    }
    for metric, source_column in metric_columns.items():
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for policy in policies:
            if policy == "dqn":
                values = [
                    float(row[summary_columns[metric]])
                    for row in per_seed_summaries
                    if row["policy"] == "dqn"
                ]
            else:
                values = [float(row[source_column]) for row in rows if row["policy"] == policy]
            means[policy] = _mean(values)
            stds[policy] = _std(values)
        direction = METRIC_DIRECTIONS[metric]
        winner = (
            max(means, key=means.get)
            if direction == "max"
            else min(means, key=means.get)
        )
        output_rows.append(
            {
                "metric": metric,
                "dqn_mean": round(means.get("dqn", 0.0), 6),
                "dqn_std": round(stds.get("dqn", 0.0), 6),
                "heuristic_mean": round(means.get("heuristic", 0.0), 6),
                "heuristic_std": round(stds.get("heuristic", 0.0), 6),
                "random_mean": round(means.get("random", 0.0), 6),
                "random_std": round(stds.get("random", 0.0), 6),
                "direction": direction,
                "winner": winner,
            }
        )
    _write_csv(path, output_rows)
    return output_rows


def _write_ablation(path: Path, models: list[tuple[int | str, object]], eval_seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def heuristic_policy(obs: object, rng: Random) -> int:
        return heuristic_action_from_state([float(value) for value in obs])

    def random_policy(obs: object, rng: Random) -> int:
        return rng.randrange(len(ACTION_NAMES))

    for reward_name in REWARD_CONFIGS:
        for seed, model in models:
            def dqn_policy(obs: object, rng: Random, loaded_model: object = model) -> int:
                action, _ = loaded_model.predict(obs, deterministic=True)
                return int(action)

            for eval_seed in eval_seeds:
                rows.append(_run_episode("dqn", dqn_policy, eval_seed, train_seed=seed, reward_config_name=reward_name))
        for policy_name, policy_fn in (("heuristic", heuristic_policy), ("random", random_policy)):
            for eval_seed in eval_seeds:
                rows.append(_run_episode(policy_name, policy_fn, eval_seed, train_seed="baseline", reward_config_name=reward_name))

    summaries: list[dict[str, object]] = []
    for reward_name in REWARD_CONFIGS:
        for policy in ("dqn", "heuristic", "random"):
            values = [
                float(row["episode_reward"])
                for row in rows
                if row["policy"] == policy and row["reward_config"] == reward_name
            ]
            if not values:
                continue
            summaries.append(
                {
                    "ablation": reward_name,
                    "policy": policy,
                    "mean_reward": round(_mean(values), 6),
                    "std_reward": round(_std(values), 6),
                    "eval_episodes": len(values),
                }
            )
    _write_csv(path, summaries)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_dqn_models(model_dir: Path, train_seeds: list[int]) -> list[tuple[int | str, object]]:
    try:
        from stable_baselines3 import DQN  # type: ignore[import-not-found]
    except Exception:
        return []

    models: list[tuple[int | str, object]] = []
    for seed in train_seeds:
        path = model_dir / f"coach_policy_seed_{seed}.zip"
        if path.exists():
            models.append((seed, DQN.load(str(path))))
    if not models:
        fallback_path = ROOT / "models" / "coach_policy.zip"
        if fallback_path.exists():
            models.append(("canonical", DQN.load(str(fallback_path))))
    return models


def _update_manifest(path: Path, command: str, comparison_rows: list[dict[str, object]]) -> None:
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    manifest["evaluation_command"] = command
    manifest["policy_comparison"] = comparison_rows
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _evaluate_policies(
    rl_dir: Path,
    *,
    model_dir: Path,
    train_seeds: list[int],
    eval_seeds: list[int],
    command: str,
) -> dict[str, Path]:
    rl_dir.mkdir(parents=True, exist_ok=True)
    models = _load_dqn_models(model_dir, train_seeds)
    rows: list[dict[str, object]] = []

    for train_seed, model in models:
        def dqn_policy(obs: object, rng: Random, loaded_model: object = model) -> int:
            action, _ = loaded_model.predict(obs, deterministic=True)
            return int(action)

        for eval_seed in eval_seeds:
            rows.append(_run_episode("dqn", dqn_policy, eval_seed, train_seed=train_seed))

    def heuristic_policy(obs: object, rng: Random) -> int:
        return heuristic_action_from_state([float(value) for value in obs])

    def random_policy(obs: object, rng: Random) -> int:
        return rng.randrange(len(ACTION_NAMES))

    for eval_seed in eval_seeds:
        rows.append(_run_episode("heuristic", heuristic_policy, eval_seed, train_seed="baseline"))
        rows.append(_run_episode("random", random_policy, eval_seed, train_seed="baseline"))

    eval_rows_path = rl_dir / "episode_eval_rows.csv"
    _write_csv(eval_rows_path, rows)
    per_seed_path = rl_dir / "per_seed_eval_metrics.csv"
    per_seed_summaries = _write_per_seed_metrics(per_seed_path, rows)
    comparison_path = rl_dir / "policy_comparison.csv"
    comparison_rows = _write_policy_comparison(comparison_path, rows, per_seed_summaries)
    ablation_path = rl_dir / "reward_ablation.csv"
    ablation_rows = _write_ablation(ablation_path, models, eval_seeds)

    reward_rows = _read_episode_rewards(rl_dir / "episode_rewards.csv")
    rewards_by_episode: dict[int, list[float]] = defaultdict(list)
    for row in reward_rows:
        rewards_by_episode[int(row["episode"])].append(row["reward"])
    curve_values = [(episode, _mean(values)) for episode, values in sorted(rewards_by_episode.items())]
    _write_svg_line_chart(
        rl_dir / "reward_curve.svg",
        "Mean DQN training reward logged by SB3 Monitor",
        "Episode",
        "Mean reward",
        curve_values,
    )

    reward_metric = next(row for row in comparison_rows if row["metric"] == "reward")
    labels = [policy for policy in ("dqn", "heuristic", "random") if any(row["policy"] == policy for row in rows)]
    values = [float(reward_metric[f"{policy}_mean"]) for policy in labels]
    _write_svg_bar_chart(rl_dir / "baseline_comparison.svg", "Held-out reward comparison", labels, values, "Mean reward")
    _write_action_distribution(rl_dir / "action_distribution.svg", rows)
    _update_manifest(rl_dir / "run_manifest.json", command, comparison_rows)

    return {
        "episode_eval_rows": eval_rows_path,
        "per_seed_eval_metrics": per_seed_path,
        "policy_comparison": comparison_path,
        "reward_ablation": ablation_path,
        "rl_reward_curve": rl_dir / "reward_curve.svg",
        "baseline_comparison": rl_dir / "baseline_comparison.svg",
        "action_distribution": rl_dir / "action_distribution.svg",
        "run_manifest": rl_dir / "run_manifest.json",
    }


def evaluate(
    output_dir: Path,
    *,
    model_dir: Path = ROOT / "models" / "rl_runs",
    train_seeds: Iterable[int] = DEFAULT_TRAIN_SEEDS,
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    command: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(ROOT / "data" / "profiles" / "default_user.json")
    analysis = analyze_landmark_csv(bundled_synthetic_landmarks(), output_dir, profile)
    outputs = make_coaching_outputs(analysis.rep_features, profile, HeuristicPolicy())
    write_coaching_outputs(outputs, output_dir / "latest_session" / "coach_outputs.json")

    _write_svg_line_chart(
        output_dir / "fatigue_over_reps.svg",
        "Fatigue over reps",
        "Rep",
        "Fatigue score",
        [(rep.rep_id, rep.fatigue_score) for rep in analysis.rep_features],
    )
    _write_svg_line_chart(
        output_dir / "injury_risk_over_reps.svg",
        "Injury risk over reps",
        "Rep",
        "Risk score",
        [(rep.rep_id, rep.injury_risk) for rep in analysis.rep_features],
    )

    mistake_counts = Counter(rep.mistake_label for rep in analysis.rep_features if rep.mistake_label != "none")
    metrics_path = output_dir / "evaluation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "detected_reps", "value": len(analysis.rep_features)})
        writer.writerow({"metric": "expected_synthetic_reps", "value": 5})
        writer.writerow({"metric": "rep_count_accuracy", "value": round(len(analysis.rep_features) / 5, 3)})
        writer.writerow({"metric": "average_fatigue", "value": round(analysis.average_fatigue, 3)})
        writer.writerow({"metric": "average_injury_risk", "value": round(analysis.average_injury_risk, 3)})
        writer.writerow({"metric": "mistake_types", "value": dict(mistake_counts)})

    rl_dir = output_dir if output_dir.name == "rl_training" else output_dir / "rl_training"
    command = command or shlex.join([sys.executable, *sys.argv])
    rl_paths = _evaluate_policies(
        rl_dir,
        model_dir=model_dir,
        train_seeds=[int(seed) for seed in train_seeds],
        eval_seeds=[int(seed) for seed in eval_seeds],
        command=command,
    )

    root_reward_curve = output_dir / "reward_curve.svg"
    root_reward_curve.write_text((rl_dir / "reward_curve.svg").read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "metrics": metrics_path,
        "reward_curve": root_reward_curve,
        "fatigue": output_dir / "fatigue_over_reps.svg",
        "risk": output_dir / "injury_risk_over_reps.svg",
        "coach_outputs": output_dir / "latest_session" / "coach_outputs.json",
        **rl_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adaptive RL Gym Coach outputs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "rl_runs")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    args = parser.parse_args()
    paths = evaluate(
        args.output_dir,
        model_dir=args.model_dir,
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        command=shlex.join([sys.executable, *sys.argv]),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
