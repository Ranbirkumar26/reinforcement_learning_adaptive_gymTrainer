# Adaptive Reinforcement Learning-Based Explainable AI Gym Coach

## Abstract

This project implements a squat-focused AI gym coach that uses human pose estimation, personalized biomechanics, reinforcement learning, and explainable feedback. The prototype accepts squat video or a synthetic landmark session, extracts pose landmarks, computes joint and movement features per repetition, estimates fatigue and injury risk, and selects a coaching action through a reinforcement-learning policy or deterministic fallback. Each recommendation includes a problem, reason, correction, and evidence frame.

## Introduction

Most AI fitness tools treat exercise coaching as posture classification. They compare a user against fixed rules and return the same correction for many body types, fatigue levels, and training histories. This project frames coaching as sequential decision-making. The system observes the current rep, user history, fatigue, and risk signals, then chooses the next coaching action.

## Problem Statement

Existing AI fitness applications often use static thresholds. They do not adapt enough to individual biomechanics, they do not account for fatigue, and they rarely remember recurring movement errors across sessions. This creates generic feedback that can miss user-specific risk and over-correct natural fatigue.

## Objectives

- Extract human pose landmarks from squat video.
- Compute per-rep biomechanics features.
- Model fatigue, injury risk, and recurring mistakes.
- Define a Markov Decision Process for coaching.
- Train or simulate a reinforcement-learning policy for coaching actions.
- Explain each coaching decision using interpretable movement evidence.
- Deliver a working Streamlit demo and submission artifacts.

## Literature Review

Human pose estimation systems such as MediaPipe Pose make markerless movement analysis feasible on consumer hardware. Biomechanics-based exercise coaching commonly uses joint angles, depth, symmetry, and torso position to evaluate exercise technique. Reinforcement learning supports sequential decision problems where an action affects future outcomes. Explainable AI increases user trust by showing why a recommendation was made instead of only labeling a movement as correct or incorrect.

## Proposed System

The system has five layers:

1. Pose estimation extracts shoulder, hip, knee, and ankle landmarks.
2. Biomechanics processing computes knee angle, hip angle, torso lean, knee tracking, depth, tempo, and smoothness.
3. Personalization compares movement against a user profile and baseline angles.
4. Coaching intelligence estimates fatigue, injury risk, and repeated mistakes.
5. The RL agent selects a coaching action and returns an explanation.

## Architecture

Input video flows through OpenCV and MediaPipe Pose when those dependencies are installed. Extracted landmarks are saved to landmarks.csv. The biomechanics layer segments squat repetitions and writes rep_features.csv. The coaching layer converts each rep into a state vector, chooses an action, computes reward, and writes coach_outputs.json. The Streamlit dashboard collects client details, runs video analysis, then enables a results view with plain-language coaching on one side and technical policy reasoning on the other.

## MDP Formulation

The state vector contains normalized knee angle, hip angle, torso lean, rep progress, fatigue score, injury risk, repeated mistake count, previous action, and skill level. The action space has eight actions: verbal cue, joint highlight, slow tempo, adjust difficulty, recommend rest, no feedback, demonstration, and breathing cue. The reward increases when future movement quality improves and decreases when risk, fatigue, repeated mistakes, or excessive corrections increase.

## Implementation

The implementation uses Python 3.11 and Streamlit. Core modules implement geometry, pose extraction, rep segmentation, biomechanics features, user profiles, RL environment, policy selection, evaluation, artifact generation, and dashboard explanations. Stable-Baselines3 DQN trains on synthetic coaching trajectories when installed. If optional RL dependencies are unavailable, the project writes a heuristic policy artifact so the demo remains usable.

## Results

The bundled synthetic session produces a complete end-to-end run. The system detects squat repetitions, calculates fatigue and injury-risk scores, selects coaching actions, and writes all declared data contracts. Evaluation generates reward, fatigue, and risk charts. The dashboard converts those outputs into two views: a trainer-style explanation for non-technical users and a technical trace showing pose features, thresholds, state vector, raw model action, final safety-filtered action, reward, and evidence frame.

The supplied real squat video was processed successfully. The pipeline wrote landmarks, rep features, coaching outputs, an overlay video, and evidence frames. It detected 17 repetitions. The top detected issue was knee tracking. The final coaching output selected joint highlight for 15 reps and rest recommendation for 2 reps.

## Testing and QA

The final QA pass used a local Python 3.12 virtual environment. Unit and integration tests covered geometry, rep segmentation, biomechanics extraction, reward scoring, action mapping, profile persistence, dashboard summary logic, technical explanation traces, malformed input, missing video paths, and the synthetic end-to-end pipeline. The full pytest suite passed with 28 tests. Streamlit responded successfully to a health check. The supplied real squat video wrote landmarks, an overlay video, evidence frames, rep features, and coaching outputs.

The only open issue is a non-blocking MediaPipe packaging metadata warning on macOS arm64. MediaPipe 0.10.21 imports correctly and exposes the required pose API, but pip check reports that the wheel is not supported on this platform.

## Explainability

Each coaching output includes a problem, reason, correction, and evidence frame. Examples include knee tracking issues caused by increased knee-offset proxy, excessive forward lean caused by torso angle drift, and fatigue signals caused by tempo slowdown and movement instability.

## Limitations

The MVP focuses only on squats. Injury risk signals are heuristic proxies and are not medical diagnosis. RL training uses simulated trajectories because real multi-session feedback data is unavailable. Camera angle, occlusion, clothing, and landmark confidence affect accuracy. Grad-CAM is deferred because this prototype does not train a CNN image classifier.

## Future Scope

Future work can add push-ups and lunges, live webcam coaching, real user feedback loops, improved clinical validation, mobile deployment, and deeper explainability with image-based models.

## Conclusion

The prototype demonstrates the main project contribution: exercise coaching can be treated as a personalized sequential decision problem. The system does more than classify posture. It learns or simulates how to choose coaching actions based on movement, fatigue, risk, and user history.

## References

- Lugaresi, C. et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv, 2019.
- Farnebäck, G. Two-Frame Motion Estimation Based on Polynomial Expansion. Scandinavian Conference on Image Analysis, 2003.
- Mnih, V. et al. Human-level control through deep reinforcement learning. Nature, 2015.
- Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv, 2017.
- Samek, W., Wiegand, T., and Müller, K. Explainable Artificial Intelligence. ITU Journal, 2017.
