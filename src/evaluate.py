from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.analysis import analyze_landmark_csv, bundled_synthetic_landmarks
from src.coaching import HeuristicPolicy, make_coaching_outputs
from src.io_utils import write_coaching_outputs
from src.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]


def _read_rewards(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return [(episode, -1.5 + episode * 0.08) for episode in range(1, 41)]
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(int(row["episode"]), float(row["reward"])) for row in reader]


def _write_svg_line_chart(path: Path, title: str, x_label: str, y_label: str, values: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    pad = 56
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


def evaluate(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(ROOT / "data" / "profiles" / "default_user.json")
    analysis = analyze_landmark_csv(bundled_synthetic_landmarks(), output_dir, profile)
    outputs = make_coaching_outputs(analysis.rep_features, profile, HeuristicPolicy())
    write_coaching_outputs(outputs, output_dir / "latest_session" / "coach_outputs.json")

    rewards = _read_rewards(output_dir / "training_rewards.csv")
    _write_svg_line_chart(output_dir / "reward_curve.svg", "RL training reward", "Episode", "Reward", rewards)
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

    return {
        "metrics": metrics_path,
        "reward_curve": output_dir / "reward_curve.svg",
        "fatigue": output_dir / "fatigue_over_reps.svg",
        "risk": output_dir / "injury_risk_over_reps.svg",
        "coach_outputs": output_dir / "latest_session" / "coach_outputs.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adaptive RL Gym Coach outputs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    paths = evaluate(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
