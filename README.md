# Adaptive RL Gym Coach

Squat-only submission MVP for a reinforcement-learning gym coach. The app extracts pose landmarks, derives biomechanics features per rep, estimates fatigue and injury risk, chooses a coaching action, and explains the correction.

## Architecture

```text
Video or sample landmarks
        |
        v
MediaPipe/OpenCV pose pipeline -> landmarks.csv -> rep segmentation
        |                                      |
        v                                      v
pose_overlay.mp4                     biomechanics features
                                               |
                                               v
fatigue, injury risk, mistake labels -> RL policy -> coach_outputs.json
                                               |
                                               v
Streamlit demo, charts, report, PPT
```

Core data contracts:

- `landmarks.csv`: frame-level pose landmarks.
- `rep_features.csv`: per-rep biomechanics, fatigue, injury risk, and mistake labels.
- `coach_outputs.json`: per-rep state, action, explanation, reward, and evidence frame.

## Quick start from GitHub

```bash
git clone https://github.com/Ranbirkumar26/reinforcement_learning_adaptive_gymTrainer.git
cd reinforcement_learning_adaptive_gymTrainer
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_sample_assets
python -m src.train_rl
python -m src.evaluate
python -m pytest -q
streamlit run app.py
```

Open Streamlit, enter client name, upload `data/sample_videos/real_squat_sample.mov` or use bundled sample checkbox, run analysis, then click `Show results`.

## Local quick start

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_sample_assets
python -m src.train_rl
python -m src.evaluate
streamlit run app.py
```

## Main Commands

```bash
python -m pytest -q
python -m compileall src app.py report/build_report.py
python -m src.generate_sample_assets
python -m src.train_rl --timesteps 200
python -m src.evaluate
python report/build_report.py
node slides/build_deck.mjs
streamlit run app.py --server.headless true
```

## What the Prototype Proves

- Pose estimation detects squat movement when MediaPipe and OpenCV are installed.
- Biomechanics features are extracted per rep.
- Fatigue, injury risk, and user history influence coaching.
- RL policy selects one of eight coaching actions.
- Each coaching action includes problem, reason, correction, and evidence frame.
- Report, PPT, charts, and demo outputs can be generated locally.

## Project Layout

```text
app.py                     Streamlit demo
src/                       Core package
data/profiles/             User profiles
data/sample_landmarks/     Synthetic squat landmark CSV
data/sample_videos/        Bundled real sample plus optional user videos
models/                    Trained DQN or heuristic policy artifact
outputs/                   Analysis outputs, charts, overlay videos
report/                    Report markdown, builder, PDF
slides/                    PPT builder and final deck
tests/                     Pytest tests for core logic
```

## Included submission artifacts

```text
data/sample_videos/real_squat_sample.mov
outputs/real_video_session/latest_session/landmarks.csv
outputs/real_video_session/latest_session/rep_features.csv
outputs/real_video_session/latest_session/coach_outputs.json
outputs/real_video_session/latest_session/pose_overlay.mp4
models/coach_policy.zip
outputs/evaluation_metrics.csv
outputs/reward_curve.svg
outputs/fatigue_over_reps.svg
outputs/injury_risk_over_reps.svg
report/Adaptive_RL_Gym_Coach_Report.pdf
slides/final_adaptive_rl_gym_coach_v3.pptx
QA_BUG_LOG.md
submission_manifest.md
```

## Demo Flow

1. Select sample profile.
2. Upload squat video, or use bundled real sample video.
3. Run analysis.
4. Click `Show results` after analysis completes.
5. Review user-side coaching explanation and technical-side model reasoning.
6. Review rep table, fatigue chart, injury-risk chart, coaching actions, overlay video, and evidence frames.

## Test status

Final verified commands:

```bash
python -m compileall src app.py report/build_report.py
python -m pytest -q
python -m src.generate_sample_assets
python -m src.train_rl --timesteps 200
python -m src.evaluate
streamlit run app.py --server.headless true
```

Current result: `28 passed`. Supplied real squat video processed successfully with 17 detected reps.

## Notes

- Real video analysis requires `opencv-python` and `mediapipe`.
- RL training uses Stable-Baselines3 when installed. If unavailable, training writes a deterministic heuristic policy artifact so the demo still runs.
- Sample landmarks are synthetic and support reproducible tests and charts.
- Included real squat sample: `data/sample_videos/real_squat_sample.mov`.
- Real-video analysis outputs are under `outputs/real_video_session/`.
- On macOS arm64, old MediaPipe `mp.solutions` wheels may trigger `pip check` platform metadata warnings even when runtime imports and pose APIs work. The tested version is `mediapipe==0.10.21`.
