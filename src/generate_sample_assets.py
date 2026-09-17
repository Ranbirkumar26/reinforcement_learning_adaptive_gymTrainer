from __future__ import annotations

from pathlib import Path

from src.io_utils import write_landmarks_csv
from src.synthetic import generate_synthetic_squat_landmarks


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    csv_path = ROOT / "data" / "sample_landmarks" / "synthetic_squat_landmarks.csv"
    rows = generate_synthetic_squat_landmarks()
    write_landmarks_csv(rows, csv_path)
    videos_readme = ROOT / "data" / "sample_videos" / "README.md"
    videos_readme.parent.mkdir(parents=True, exist_ok=True)
    videos_readme.write_text(
        "Place short squat videos here for demo runs. Use the bundled synthetic landmark CSV when no video is available.\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {videos_readme}")


if __name__ == "__main__":
    main()
