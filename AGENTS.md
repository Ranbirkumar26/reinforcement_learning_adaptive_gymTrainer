# Adaptive RL Gym Coach

## Decision Log

- 2026-09-17: MVP targets squat coaching only. Reason: one complete exercise is stronger for submission than partial multi-exercise support.
- 2026-09-17: Streamlit chosen for demo UI. Reason: fast local demo, no paid hosting, simple video upload flow.
- 2026-09-17: MediaPipe Pose chosen for pose extraction. Reason: free local pose estimation with established landmark schema.
- 2026-09-17: Stable-Baselines3 DQN chosen for RL when installed. Reason: simple discrete action policy matching eight coaching actions.
- 2026-09-17: Synthetic trajectories allowed for RL training. Reason: real multi-session user feedback data is not available at project start.
- 2026-09-17: Explainability uses biomechanical reasons, risk scores, and highlighted skeleton frames. Reason: Grad-CAM requires a CNN image model outside MVP scope.

## Deferred Features

- "push-up/lunge support"
- "real mobile app"
- "production auth"
- "live webcam coaching"
- "true clinical injury prediction"
- "Grad-CAM CNN pipeline"

## Submission Notes

- Keep secrets out of code and logs. Current project uses no paid APIs and no external keys.
- Use local sample videos or user-recorded squat videos. Public videos need citation in the report.
- If dependency installation is constrained, core logic and tests still run on synthetic sessions.
