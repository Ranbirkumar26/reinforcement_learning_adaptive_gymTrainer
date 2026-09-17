# Submission Manifest

## Required Files

- Source code: `src/`, `app.py`
- Setup: `requirements.txt`, `README.md`
- Test config: `pyproject.toml`
- QA log: `QA_BUG_LOG.md`
- Project continuity: `AGENTS.md`
- User profile: `data/profiles/default_user.json`
- Sample input fallback: `data/sample_landmarks/synthetic_squat_landmarks.csv`
- Video input folder: `data/sample_videos/`
- Real squat sample: `data/sample_videos/real_squat_sample.mov`
- Policy artifact: `models/coach_policy.zip`
- Fallback policy metadata: `models/coach_policy.json`
- Evaluation outputs: `outputs/`
- Real video outputs: `outputs/real_video_session/`
- Report PDF: `report/Adaptive_RL_Gym_Coach_Report.pdf`
- Final PPT: `slides/final_adaptive_rl_gym_coach_v3.pptx`
- Tests: `tests/`

## Recommended Demo Order

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_sample_assets
python -m src.train_rl
python -m src.evaluate
streamlit run app.py
```

## Current QA Status

- Synthetic landmark sample generated.
- Stable-Baselines3 policy artifact generated in `.venv`.
- Evaluation metrics and SVG charts generated.
- Report PDF generated and visually checked.
- Final PPT generated, validated, and rendered for visual check.
- `pytest` passed: 22 tests.
- `compileall` passed.
- `python -m src.train_rl --timesteps 200` passed and wrote `models/coach_policy.zip`.
- `python -m src.evaluate` passed and refreshed charts plus metrics.
- Streamlit health check passed.
- MediaPipe imports and `mp.solutions.pose` exists.
- `pip check` has one open P3 metadata issue: `mediapipe 0.10.21 is not supported on this platform`.
- Real squat video pipeline passed on `data/sample_videos/real_squat_sample.mov`.
- Real video outputs include landmarks CSV, rep features CSV, coaching JSON, overlay video, and evidence frames.
- Real video detected 17 reps.
- Real video top mistake: knee tracking.
- Real video coaching actions: 15 joint highlights, 2 rest recommendations.
- No open P0, P1, or P2 bugs remain.
