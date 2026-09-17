from __future__ import annotations

from random import Random

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional deps missing
    gym = None
    np = None
    spaces = None

from src.actions import ACTION_NAMES


class SyntheticSquatCoachEnv(gym.Env if gym else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 12, seed: int = 11) -> None:
        self.episode_length = episode_length
        self.rng = Random(seed)
        self.step_index = 0
        self.previous_action = ACTION_NAMES.index("no_feedback")
        self.fatigue = 0.0
        self.risk = 0.2
        self.repeated = 0
        if spaces is not None:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(ACTION_NAMES))

    def _state(self) -> list[float]:
        progress = self.step_index / max(1, self.episode_length)
        knee = max(0.35, 0.78 - 0.15 * progress + self.rng.uniform(-0.04, 0.04))
        hip = max(0.30, 0.72 - 0.12 * progress + self.rng.uniform(-0.03, 0.03))
        torso = min(1.0, 0.24 + 0.18 * progress + self.rng.uniform(-0.04, 0.04))
        return [
            knee,
            hip,
            torso,
            progress,
            self.fatigue,
            self.risk,
            min(1.0, self.repeated / 5),
            self.previous_action / max(1, len(ACTION_NAMES) - 1),
            0.0,
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if seed is not None:
            self.rng.seed(seed)
        self.step_index = 0
        self.previous_action = ACTION_NAMES.index("no_feedback")
        self.fatigue = self.rng.uniform(0.05, 0.18)
        self.risk = self.rng.uniform(0.12, 0.30)
        self.repeated = 0
        state = self._state()
        if np is not None:
            return np.array(state, dtype=np.float32), {}
        return state, {}

    def step(self, action: int):  # type: ignore[override]
        good_action = (
            (self.fatigue > 0.65 and ACTION_NAMES[action] == "recommend_rest")
            or (self.risk > 0.55 and ACTION_NAMES[action] == "joint_highlight")
            or (self.fatigue <= 0.35 and self.risk <= 0.35 and ACTION_NAMES[action] == "no_feedback")
        )
        reward = 0.35 if good_action else -0.18
        if ACTION_NAMES[action] != "no_feedback":
            reward -= 0.04
        self.fatigue = min(1.0, self.fatigue + self.rng.uniform(0.02, 0.11))
        self.risk = min(1.0, max(0.0, self.risk + self.rng.uniform(-0.05, 0.10) - (0.08 if good_action else 0.0)))
        self.repeated = self.repeated + (0 if good_action else 1)
        self.previous_action = action
        self.step_index += 1
        terminated = self.step_index >= self.episode_length
        state = self._state()
        if np is not None:
            return np.array(state, dtype=np.float32), reward, terminated, False, {}
        return state, reward, terminated, False, {}
