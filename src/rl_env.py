from __future__ import annotations

from dataclasses import dataclass
from random import Random

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional deps missing
    gym = None
    np = None
    spaces = None

from src.actions import ACTION_NAMES, action_id, action_name


STATE_SIZE = 9
ACTION_COUNT = len(ACTION_NAMES)


@dataclass(frozen=True)
class RewardConfig:
    name: str = "full_reward"
    include_fatigue_penalty: bool = True
    include_risk_penalty: bool = True
    include_repeated_mistake_penalty: bool = True
    include_correction_penalty: bool = True


REWARD_CONFIGS: dict[str, RewardConfig] = {
    "full_reward": RewardConfig(),
    "no_fatigue_penalty": RewardConfig(name="no_fatigue_penalty", include_fatigue_penalty=False),
    "no_risk_penalty": RewardConfig(name="no_risk_penalty", include_risk_penalty=False),
    "no_repeated_mistake_penalty": RewardConfig(
        name="no_repeated_mistake_penalty",
        include_repeated_mistake_penalty=False,
    ),
    "no_correction_penalty": RewardConfig(name="no_correction_penalty", include_correction_penalty=False),
}


def get_reward_config(name: str) -> RewardConfig:
    try:
        return REWARD_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(REWARD_CONFIGS))
        raise ValueError(f"Unknown reward config '{name}'. Available: {available}") from exc


def heuristic_action_from_state(state: list[float]) -> int:
    fatigue = state[4]
    risk = state[5]
    repeated = int(round(state[6] * 5))
    torso = state[2]
    knee = state[0]

    if fatigue >= 0.68:
        return action_id("recommend_rest")
    if risk >= 0.56:
        return action_id("joint_highlight")
    if repeated >= 3:
        return action_id("demonstration")
    if torso >= 0.58:
        return action_id("slow_tempo")
    if knee >= 0.64 and repeated >= 1:
        return action_id("adjust_difficulty")
    if fatigue >= 0.48:
        return action_id("breathing_cue")
    if repeated >= 1:
        return action_id("verbal_cue")
    return action_id("no_feedback")


class SyntheticSquatCoachEnv(gym.Env if gym else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_length: int = 12,
        seed: int = 11,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.episode_length = episode_length
        self.initial_seed = seed
        self.reward_config = reward_config or REWARD_CONFIGS["full_reward"]
        self.rng = Random(seed)
        self.step_index = 0
        self.previous_action = action_id("no_feedback")
        self.fatigue = 0.0
        self.risk = 0.2
        self.repeated = 0
        self.movement_quality = 0.72
        self.corrections = 0
        self.skill_encoding = 0.0
        self.current_state: list[float] | None = None
        if spaces is not None:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32)
            self.action_space = spaces.Discrete(ACTION_COUNT)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _state(self) -> list[float]:
        progress = self.step_index / max(1, self.episode_length)
        form_loss = (1.0 - self.movement_quality) * 0.22
        knee = self._clamp(0.76 - 0.12 * progress + form_loss + self.rng.uniform(-0.035, 0.035), 0.30, 0.95)
        hip = self._clamp(0.70 - 0.10 * progress + form_loss + self.rng.uniform(-0.030, 0.030), 0.28, 0.95)
        torso = self._clamp(0.24 + 0.26 * self.risk + 0.16 * self.fatigue + self.rng.uniform(-0.030, 0.030))
        return [
            knee,
            hip,
            torso,
            self._clamp(progress),
            self._clamp(self.fatigue),
            self._clamp(self.risk),
            self._clamp(self.repeated / 5),
            self.previous_action / max(1, ACTION_COUNT - 1),
            self.skill_encoding,
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if seed is not None:
            self.rng.seed(seed)
        self.step_index = 0
        self.previous_action = action_id("no_feedback")
        self.fatigue = self.rng.uniform(0.05, 0.22)
        self.risk = self.rng.uniform(0.12, 0.34)
        self.repeated = self.rng.choice([0, 0, 1])
        self.movement_quality = self.rng.uniform(0.66, 0.86)
        self.corrections = 0
        self.skill_encoding = self.rng.choice([0.0, 0.5, 1.0])
        state = self._state()
        self.current_state = state
        if np is not None:
            return np.array(state, dtype=np.float32), {}
        return state, {}

    def _quality_score(self) -> float:
        return self.movement_quality - self.risk * 0.50 - self.fatigue * 0.25 - self.repeated * 0.08

    def _apply_action_effect(self, action: int, preferred_action: int) -> None:
        name = action_name(action)
        matched = action == preferred_action
        fatigue_drift = 0.035 + self.rng.uniform(0.000, 0.045)
        risk_drift = -0.012 + self.rng.uniform(0.000, 0.050) + self.repeated * 0.006
        quality_drift = -0.020 - self.fatigue * 0.018 + self.rng.uniform(-0.020, 0.020)

        self.fatigue = self._clamp(self.fatigue + fatigue_drift)
        self.risk = self._clamp(self.risk + risk_drift)
        self.movement_quality = self._clamp(self.movement_quality + quality_drift)

        if matched:
            if name == "recommend_rest":
                self.fatigue = self._clamp(self.fatigue - 0.22)
                self.risk = self._clamp(self.risk - 0.035)
                self.movement_quality = self._clamp(self.movement_quality + 0.040)
            elif name == "joint_highlight":
                self.risk = self._clamp(self.risk - 0.130)
                self.movement_quality = self._clamp(self.movement_quality + 0.060)
            elif name == "demonstration":
                self.risk = self._clamp(self.risk - 0.060)
                self.movement_quality = self._clamp(self.movement_quality + 0.080)
                self.repeated = max(0, self.repeated - 2)
            elif name == "slow_tempo":
                self.fatigue = self._clamp(self.fatigue - 0.070)
                self.risk = self._clamp(self.risk - 0.055)
                self.movement_quality = self._clamp(self.movement_quality + 0.050)
            elif name == "adjust_difficulty":
                self.fatigue = self._clamp(self.fatigue - 0.060)
                self.risk = self._clamp(self.risk - 0.080)
                self.movement_quality = self._clamp(self.movement_quality + 0.055)
            elif name == "breathing_cue":
                self.fatigue = self._clamp(self.fatigue - 0.055)
                self.risk = self._clamp(self.risk - 0.025)
                self.movement_quality = self._clamp(self.movement_quality + 0.030)
            elif name == "verbal_cue":
                self.risk = self._clamp(self.risk - 0.025)
                self.movement_quality = self._clamp(self.movement_quality + 0.040)
            elif name == "no_feedback":
                self.movement_quality = self._clamp(self.movement_quality + 0.025)
            self.repeated = max(0, self.repeated - 1)
            return

        if name == "no_feedback":
            self.risk = self._clamp(self.risk + 0.060)
            self.movement_quality = self._clamp(self.movement_quality - 0.065)
            self.repeated += 1
            return

        self.fatigue = self._clamp(self.fatigue + 0.025)
        self.risk = self._clamp(self.risk + 0.020)
        self.movement_quality = self._clamp(self.movement_quality - 0.020)
        if self.rng.random() < 0.45:
            self.repeated += 1

    def _reward_terms(self, before_quality: float, action: int, preferred_action: int) -> dict[str, float]:
        after_quality = self._quality_score()
        action_name_value = action_name(action)
        matched = action == preferred_action
        terms = {
            "improvement": max(-0.55, min(0.75, after_quality - before_quality)) * 1.6,
            "action_match": 0.28 if matched else -0.16,
            "fatigue_penalty": -0.20 * self.fatigue if self.reward_config.include_fatigue_penalty else 0.0,
            "risk_penalty": -0.36 * self.risk if self.reward_config.include_risk_penalty else 0.0,
            "repeated_mistake_penalty": (
                -min(0.55, self.repeated * 0.12)
                if self.reward_config.include_repeated_mistake_penalty
                else 0.0
            ),
            "correction_penalty": (
                -0.055
                if self.reward_config.include_correction_penalty and action_name_value != "no_feedback"
                else 0.0
            ),
        }
        return terms

    def step(self, action: int):  # type: ignore[override]
        if action < 0 or action >= ACTION_COUNT:
            raise ValueError(f"Invalid action id: {action}")
        state_before = self.current_state if self.current_state is not None else self._state()
        preferred_action = heuristic_action_from_state(state_before)
        before_quality = self._quality_score()
        if action_name(action) != "no_feedback":
            self.corrections += 1
        self._apply_action_effect(action, preferred_action)
        terms = self._reward_terms(before_quality, action, preferred_action)
        reward = sum(terms.values())
        self.previous_action = action
        self.step_index += 1
        terminated = self.step_index >= self.episode_length
        state = self._state()
        self.current_state = state
        info = {
            "reward_config": self.reward_config.name,
            "reward_terms": terms,
            "preferred_action": action_name(preferred_action),
            "selected_action": action_name(action),
            "final_fatigue": self.fatigue,
            "final_risk": self.risk,
            "final_repeated_mistakes": float(self.repeated),
            "correction_rate": self.corrections / max(1, self.step_index),
        }
        if np is not None:
            return np.array(state, dtype=np.float32), float(reward), terminated, False, info
        return state, float(reward), terminated, False, info
