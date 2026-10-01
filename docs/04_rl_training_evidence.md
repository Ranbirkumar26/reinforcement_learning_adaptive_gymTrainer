# RL Training Evidence

## Purpose

This document records reproducible reinforcement-learning evidence for the Adaptive RL Gym Coach MVP. It is written for review-panel questions about machine learning, deep learning, and reinforcement learning.

The goal is evidence, not model marketing. The DQN result is reported exactly as measured. The reward function, training seeds, evaluation seeds, and metrics were fixed before running final experiments. No reward weights, seeds, or evaluation settings were changed after seeing results.

## Scope Boundary

| Topic | What this project proves | What it does not claim |
|---|---|---|
| RL training | DQN can be trained on a deterministic squat-coaching simulator and evaluated across fixed seeds. | It does not prove real human learning from long-term user feedback. |
| Real squat video | The real video validates pose extraction, rep segmentation, feature extraction, policy inference, explanations, overlay video, and dashboard path. | It is not used as RL training data. |
| Injury risk | Risk is a biomechanical proxy from torso lean, knee tracking, fatigue, and visible asymmetry. | It is not clinical diagnosis. |
| Explainability | Each action has state values, threshold triggers, model action, safety decision, reward, and evidence frame. | It is not Grad-CAM because no CNN posture classifier is trained. |

## Fixed Experiment Design

### Training Seeds

```text
13, 17, 23, 29, 31
```

### Held-out Evaluation Seeds

```text
1001 through 1030
```

### Training Budget

```text
50,000 timesteps per seed
5 training seeds
250,000 total DQN timesteps
```

### State Vector

The public state vector stayed backward compatible with dashboard and tests.

| Index | State value | Meaning |
|---:|---|---|
| 0 | normalized knee angle | Squat depth and knee movement signal |
| 1 | normalized hip angle | Hip range signal |
| 2 | normalized torso angle | Forward lean signal |
| 3 | rep progress | Position inside simulated set |
| 4 | fatigue score | Estimated fatigue in current session |
| 5 | injury risk score | Estimated movement risk |
| 6 | repeated mistake count | Recurring form issue memory |
| 7 | previous action id | Last coaching action |
| 8 | skill level encoding | Beginner, intermediate, advanced context |

### Action Space

Action count stayed fixed at eight.

| Action id | Action |
|---:|---|
| 0 | `verbal_cue` |
| 1 | `joint_highlight` |
| 2 | `slow_tempo` |
| 3 | `adjust_difficulty` |
| 4 | `recommend_rest` |
| 5 | `no_feedback` |
| 6 | `demonstration` |
| 7 | `breathing_cue` |

## Simulator Dynamics

Each simulator step represents one squat repetition. The environment starts from seed-controlled fatigue, risk, movement quality, repeated mistakes, and skill level. Action choice affects next-step fatigue, risk, movement quality, and repeated mistake count.

The heuristic policy is also used as an interpretable reference policy:

| Condition | Heuristic action |
|---|---|
| High fatigue | `recommend_rest` |
| High risk | `joint_highlight` |
| Repeated mistakes | `demonstration` |
| High torso lean | `slow_tempo` |
| Shallow pattern with repeated issue | `adjust_difficulty` |
| Moderate fatigue | `breathing_cue` |
| Single repeated issue | `verbal_cue` |
| Low-risk clean state | `no_feedback` |

## Reward Function

Reward combines improvement and penalties:

```text
reward =
  improvement in movement quality
  + action-match reward
  - fatigue penalty
  - risk penalty
  - repeated-mistake penalty
  - correction penalty
```

The default reward config is `full_reward`. Ablations remove one penalty at a time:

| Ablation | Removed component |
|---|---|
| `full_reward` | None |
| `no_fatigue_penalty` | Fatigue penalty |
| `no_risk_penalty` | Risk penalty |
| `no_repeated_mistake_penalty` | Repeated mistake penalty |
| `no_correction_penalty` | Correction penalty |

## Exact Commands

These commands were run from `/Users/tarry/Desktop/RL_gym_coach` inside `.venv`.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.train_rl --timesteps 50000 --seeds 13 17 23 29 31 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --output-dir outputs/rl_training --model-dir models/rl_runs --output models/coach_policy
.venv/bin/python -m src.evaluate --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --train-seeds 13 17 23 29 31 --output-dir outputs --model-dir models/rl_runs
```

The exact executed command strings and environment metadata are also stored in:

```text
outputs/rl_training/run_manifest.json
```

## Runtime Environment

| Item | Value |
|---|---|
| Git commit before this change set | `71e6437` |
| Python | `3.12.14` |
| Platform | `macOS-26.5.1-arm64-arm-64bit` |
| Stable-Baselines3 | `2.3.2` |
| Gymnasium | `0.29.1` |
| NumPy | `1.26.4` |
| pandas | `2.2.2` |
| Matplotlib | `3.9.2` |
| PyTorch | `2.14.0` |
| OpenCV | `4.10.0.84` |
| MediaPipe | `0.10.21` |
| Streamlit | `1.38.0` |

## Generated Artifacts

| Artifact | Meaning |
|---|---|
| `outputs/rl_training/episode_rewards.csv` | Real SB3 Monitor episode rewards from simulator training. 20,830 rows. |
| `outputs/rl_training/per_seed_eval_metrics.csv` | Held-out metrics per DQN seed plus heuristic and random baselines. |
| `outputs/rl_training/policy_comparison.csv` | Metric-level comparison with winner labels. |
| `outputs/rl_training/reward_ablation.csv` | Reward ablation table. |
| `outputs/rl_training/reward_curve.svg` | Reward curve from logged training episodes. |
| `outputs/rl_training/baseline_comparison.svg` | DQN vs heuristic vs random reward chart. |
| `outputs/rl_training/action_distribution.svg` | Action frequencies on held-out simulator seeds. |
| `outputs/rl_training/run_manifest.json` | Commands, seeds, versions, platform, and manifest metadata. |
| `models/rl_runs/coach_policy_seed_*.zip` | One trained DQN model per fixed seed. |
| `models/coach_policy.zip` | Canonical dashboard policy from first training seed, not best seed. |

## Policy Comparison

Lower is better for fatigue, risk, repeated mistake rate, and correction rate. Higher is better for reward.

DQN standard deviation is computed across the five training-seed means. Heuristic and random standard deviations are computed across the 30 held-out evaluation episodes.

| Metric | Direction | DQN mean | DQN std | Heuristic mean | Heuristic std | Random mean | Random std | Winner |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Reward | max | 0.714965 | 0.077572 | 0.871716 | 0.336870 | -10.769327 | 2.742853 | heuristic |
| Final fatigue | min | 0.535816 | 0.004245 | 0.528991 | 0.025421 | 0.882249 | 0.120815 | heuristic |
| Final risk | min | 0.235068 | 0.003301 | 0.234774 | 0.071252 | 0.715206 | 0.162851 | heuristic |
| Repeated mistake rate | min | 0.000556 | 0.001242 | 0.000000 | 0.000000 | 0.375000 | 0.195434 | heuristic |
| Correction rate | min | 0.440000 | 0.016736 | 0.427778 | 0.068276 | 0.897222 | 0.092027 | heuristic |

Result: DQN beats random on every metric, but the deterministic heuristic wins every measured metric on this simulator.

## Per-seed DQN Results

| Policy | Train seed | Eval episodes | Mean reward | Std reward | Mean final fatigue | Mean final risk | Mean repeated mistake rate | Mean correction rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | 13 | 30 | 0.654531 | 0.496267 | 0.540875 | 0.231814 | 0.000000 | 0.455556 |
| DQN | 17 | 30 | 0.640192 | 0.446013 | 0.539374 | 0.234180 | 0.000000 | 0.458333 |
| DQN | 23 | 30 | 0.769933 | 0.385033 | 0.534015 | 0.232862 | 0.000000 | 0.438889 |
| DQN | 29 | 30 | 0.820604 | 0.423847 | 0.530491 | 0.236341 | 0.000000 | 0.425000 |
| DQN | 31 | 30 | 0.689565 | 0.591240 | 0.534324 | 0.240141 | 0.002778 | 0.422222 |
| Heuristic | baseline | 30 | 0.871716 | 0.336870 | 0.528991 | 0.234774 | 0.000000 | 0.427778 |
| Random | baseline | 30 | -10.769327 | 2.742853 | 0.882249 | 0.715206 | 0.375000 | 0.897222 |

## Reward Ablation

| Ablation | Policy | Mean reward | Std reward | Eval episodes |
|---|---|---:|---:|---:|
| `full_reward` | DQN | 0.714965 | 0.472602 | 150 |
| `full_reward` | heuristic | 0.871716 | 0.336870 | 30 |
| `full_reward` | random | -10.769327 | 2.742853 | 30 |
| `no_fatigue_penalty` | DQN | 1.710659 | 0.440999 | 150 |
| `no_fatigue_penalty` | heuristic | 1.861043 | 0.318853 | 30 |
| `no_fatigue_penalty` | random | -9.365112 | 2.664681 | 30 |
| `no_risk_penalty` | DQN | 1.815331 | 0.340046 | 150 |
| `no_risk_penalty` | heuristic | 1.969429 | 0.169651 | 30 |
| `no_risk_penalty` | random | -8.783970 | 2.429781 | 30 |
| `no_repeated_mistake_penalty` | DQN | 0.730965 | 0.449754 | 150 |
| `no_repeated_mistake_penalty` | heuristic | 0.871716 | 0.336870 | 30 |
| `no_repeated_mistake_penalty` | random | -7.320327 | 1.373554 | 30 |
| `no_correction_penalty` | DQN | 1.005365 | 0.458146 | 150 |
| `no_correction_penalty` | heuristic | 1.154049 | 0.331344 | 30 |
| `no_correction_penalty` | random | -10.177161 | 2.744822 | 30 |

## Interpretation For Review Panel

### Machine Learning

The project uses supervised-style feature engineering from pose landmarks, then policy learning over those features. No labeled real-world dataset is invented. Synthetic trajectories provide controlled states and rewards for RL proof.

### Deep Learning

MediaPipe Pose supplies learned pose landmarks, but this project does not train MediaPipe. Stable-Baselines3 DQN uses a neural Q-network to estimate action values for the eight coaching actions.

### Reinforcement Learning

The RL setup is a Markov Decision Process:

- State: 9 movement and profile values.
- Action: 8 coaching actions.
- Transition: simulator updates fatigue, risk, repeated mistakes, and movement quality.
- Reward: improvement minus safety and over-correction penalties.
- Policy: DQN trained for fixed timesteps across fixed seeds.

### Honest Result

The heuristic policy won every metric on the current simulator. This is acceptable evidence because it shows DQN training was run and measured honestly, not tuned to claim a false win. DQN still clearly improves over random, which supports that it learned useful action preferences, but it has not surpassed the hand-designed safety heuristic.

## Reproducibility Checks

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.train_rl --timesteps 50000 --seeds 13 17 23 29 31 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --output-dir outputs/rl_training --model-dir models/rl_runs --output models/coach_policy
.venv/bin/python -m src.evaluate --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 --train-seeds 13 17 23 29 31 --output-dir outputs --model-dir models/rl_runs
```

Expected verification:

- `outputs/rl_training/episode_rewards.csv` exists and contains real training episode rows.
- `outputs/rl_training/policy_comparison.csv` contains DQN, heuristic, and random comparisons.
- `outputs/rl_training/reward_ablation.csv` contains all five reward ablations.
- `outputs/rl_training/run_manifest.json` records seeds, commands, versions, and platform.
- `pytest` passes.
