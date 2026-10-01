# QA Bug Log

## Environment

- Date: 2026-09-18
- Host: macOS arm64
- Test venv: `.venv`
- Python: 3.12.14
- Dependency install: `python -m venv .venv`, `pip install -r requirements.txt`

## Bugs

| ID | Severity | Failing test / command | Exact error | Root cause | Fix | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUG-001 | P1 | `.venv/bin/pytest -q` | `ModuleNotFoundError: No module named 'src'` | Console-script pytest did not put repo root on import path. | Added `pyproject.toml` with pytest `pythonpath = ["."]`. | `.venv/bin/pytest -q` progressed past collection. | Fixed |
| BUG-002 | P1 | `.venv/bin/python -m pytest -q` | `assert 0 == 2` in `test_segment_reps_counts_synthetic_cycles` | Rep segmentation rejected sparse sampled knee series because it required `len(knee_angles) >= fps`. | Changed early guard to require at least 3 samples. | `tests/test_biomechanics.py::test_segment_reps_counts_synthetic_cycles` passed, full pytest passed. | Fixed |
| BUG-003 | P3 | `.venv/bin/python -m pip check` | `mediapipe 0.10.21 is not supported on this platform` | MediaPipe old `mp.solutions` wheel metadata reports x86_64 on this macOS arm64 host. Runtime import works. Newer `mediapipe==0.10.35` clears metadata but removes `mp.solutions`, breaking current pose code. | Updated requirement from 0.10.14 to 0.10.21, kept old API, documented as non-blocking upstream packaging issue. | `import mediapipe as mp`, `mp.__version__ == "0.10.21"`, and `hasattr(mp.solutions, "pose")` passed. | Open P3 |
| BUG-004 | P2 | `test_missing_video_path_raises_clear_error` | Missing video path loaded optional MediaPipe dependencies before checking file existence. | File validation happened after dependency import. | Added an existence check before optional imports and tightened regression test. | `tests/test_negative_inputs.py::test_missing_video_path_raises_clear_error` passed. | Fixed |
| BUG-005 | P2 | Real squat video path test | Provided squat video initially had not been processed. | Real-video verification was pending input. | Ran `/Users/tarry/Downloads/e8994a4c-7f2f-424e-b641-751ed98843e2.mov` through MediaPipe pipeline and copied it to `data/sample_videos/real_squat_sample.mov`. | Wrote real-video landmarks, overlay video, evidence frames, 17 rep features, and coaching JSON. | Fixed |
| BUG-006 | P2 | Real-video coaching output | First high-risk reps received `no_feedback`. | DQN policy can return unsafe actions on out-of-distribution real video states. | Wrapped loaded SB3 policy in a safety policy that overrides high fatigue, high risk, and known mistake states with deterministic safe coaching actions. | `test_safety_policy_overrides_unsafe_no_feedback` passed. Real-video rerun selected 15 joint highlights and 2 rest recommendations. | Fixed |
| BUG-007 | P2 | Evidence-frame visual check | Evidence frames showed raw video without skeleton overlay. | `analyze_video` extracted evidence frames from original video rather than rendered overlay video. | Extract evidence frames from overlay video when available. | Real-video rerun produced evidence frames with skeleton overlay. | Fixed |
| BUG-008 | P2 | Streamlit pose overlay video player | Browser video player showed a dark overlay area while evidence frames rendered. | OpenCV wrote overlay MP4 as `mp4v` MPEG-4, which Streamlit/browser playback did not display reliably. | Added ffmpeg transcode step after OpenCV overlay generation: H.264 `avc1`, `yuv420p`, faststart. | `ffprobe` shows `codec_name=h264`, `codec_tag_string=avc1`, `pix_fmt=yuv420p`; dashboard restarted on port 8501. | Fixed |
| BUG-009 | P2 | `pytest tests/test_integration_pipeline.py::test_evaluate_writes_required_outputs -q` | `NameError: name 'Random' is not defined` | New evaluator policy runner used seeded random baseline without importing `Random`. | Added `from random import Random` to `src/evaluate.py`. | Targeted test passed. | Fixed |
| BUG-010 | P2 | `.venv/bin/python -m pytest tests/test_rl_evidence.py -q` | `ValueError: ... is not in the subpath of '/Users/tarry/Desktop/RL_gym_coach'` | Training manifest tried to render temp model paths as repo-relative paths during pytest. | Added safe path rendering that falls back to absolute paths outside repo. | Targeted `.venv` test passed. | Fixed |

## Final Verification Results

Passed:

- `.venv/bin/python -m compileall src app.py report/build_report.py`
- `.venv/bin/pytest -q`: 34 passed
- Fresh Python 3.11 temp venv install, compile, and pytest: 34 passed
- `.venv/bin/python -m src.generate_sample_assets`
- `.venv/bin/python -m src.train_rl --timesteps 50000 --seeds 13 17 23 29 31 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030`
- `.venv/bin/python -m src.evaluate --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --train-seeds 13 17 23 29 31`
- RL evidence: DQN mean reward 0.714965, heuristic 0.871716, random -10.769327. Heuristic won all policy-comparison metrics.
- `python report/build_report.py` regenerated `report/Adaptive_RL_Gym_Coach_Report.pdf`.
- `pandoc` regenerated review PDFs under `output/pdf/`, including `04_rl_training_evidence.pdf`.
- `node slides/build_deck.mjs` regenerated and validated 13-slide PPT.
- `pdfinfo report/Adaptive_RL_Gym_Coach_Report.pdf` passed with 4 pages.
- `pdfinfo output/pdf/04_rl_training_evidence.pdf` passed with 7 pages.
- `zip -r -FS Adaptive_RL_Gym_Coach_Submission.zip ...` refreshed final submission package and removed stale legacy reward CSV from archive.
- Streamlit health check: `STREAMLIT_HEALTH_OK 200 ok`
- Dashboard browser QA: `Show results` disabled before analysis, enabled after analysis, user-side and technical-side results rendered, first technical rep trace expanded successfully
- MediaPipe runtime import: `mediapipe 0.10.21 solutions_pose True`
- Real squat video pipeline: 17 reps detected, landmarks CSV written, rep features CSV written, coaching JSON written, overlay video written, evidence frames with skeleton overlay written

Known non-blocking issue:

- `.venv/bin/python -m pip check`: `mediapipe 0.10.21 is not supported on this platform`
- Root cause: upstream MediaPipe wheel metadata mismatch on macOS arm64 for versions that still expose `mp.solutions.pose`
- Status: P3 only. Runtime import and pose API verified.

No open P0, P1, or P2 bugs remain.
