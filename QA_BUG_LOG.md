# QA Bug Log

## Environment

- Date: 2026-09-17
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

## Final Verification Results

Passed:

- `.venv/bin/python -m compileall src app.py report/build_report.py`
- `.venv/bin/pytest -q`: 22 passed
- `.venv/bin/python -m src.generate_sample_assets`
- `.venv/bin/python -m src.train_rl --timesteps 200`
- `.venv/bin/python -m src.evaluate`
- Streamlit health check: `STREAMLIT_HEALTH_OK 200 ok`
- MediaPipe runtime import: `mediapipe 0.10.21 solutions_pose True`
- Real squat video pipeline: 17 reps detected, landmarks CSV written, rep features CSV written, coaching JSON written, overlay video written, evidence frames with skeleton overlay written

Known non-blocking issue:

- `.venv/bin/python -m pip check`: `mediapipe 0.10.21 is not supported on this platform`
- Root cause: upstream MediaPipe wheel metadata mismatch on macOS arm64 for versions that still expose `mp.solutions.pose`
- Status: P3 only. Runtime import and pose API verified.

No open P0, P1, or P2 bugs remain.
