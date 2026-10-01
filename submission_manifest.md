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
- Seeded DQN policies: `models/rl_runs/coach_policy_seed_*.zip`
- Fallback policy metadata: `models/coach_policy.json`
- Evaluation outputs: `outputs/`
- RL evidence outputs: `outputs/rl_training/`
- RL evidence doc: `docs/04_rl_training_evidence.md`
- Review-panel PDFs: `output/pdf/01_tech_stack_and_reasoning.pdf`, `output/pdf/02_algorithms_models_datasets.pdf`, `output/pdf/03_application_flow.pdf`, `output/pdf/04_rl_training_evidence.pdf`
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

For final RL evidence, run:

```bash
python -m src.train_rl --timesteps 50000 --seeds 13 17 23 29 31 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030
python -m src.evaluate --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --train-seeds 13 17 23 29 31
```

## Current QA Status

- Synthetic landmark sample generated.
- Stable-Baselines3 policy artifact generated in `.venv`.
- Five fixed-seed DQN policy artifacts generated in `.venv`.
- Evaluation metrics, RL evidence CSVs, and SVG charts generated.
- Review-panel markdown PDFs regenerated.
- Report PDF generated and checked with `pdfinfo`.
- Final PPT generated and passed package, layout, and native-chart validation.
- `pytest` passed: 34 tests.
- Fresh Python 3.11 temp venv install passed compile and pytest: 34 tests.
- `compileall` passed.
- `python -m src.train_rl --timesteps 50000 --seeds 13 17 23 29 31 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030` passed and wrote `models/coach_policy.zip` plus seed models.
- `python -m src.evaluate --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --train-seeds 13 17 23 29 31` passed and refreshed charts plus metrics.
- RL result: DQN mean reward 0.714965, heuristic 0.871716, random -10.769327. Heuristic won reward, final fatigue, final risk, repeated mistake rate, and correction rate.
- Streamlit health check passed.
- MediaPipe imports and `mp.solutions.pose` exists.
- `pip check` has one open P3 metadata issue: `mediapipe 0.10.21 is not supported on this platform`.
- Real squat video pipeline passed on `data/sample_videos/real_squat_sample.mov`.
- Real video outputs include landmarks CSV, rep features CSV, coaching JSON, overlay video, and evidence frames.
- Real video detected 17 reps.
- Real video top mistake: knee tracking.
- Real video coaching actions: 15 joint highlights, 2 rest recommendations.
- No open P0, P1, or P2 bugs remain.
