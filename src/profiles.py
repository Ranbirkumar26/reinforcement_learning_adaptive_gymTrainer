from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SkillLevel = Literal["beginner", "intermediate", "advanced"]


class UserProfile(BaseModel):
    user_id: str
    skill_level: SkillLevel = "beginner"
    height_cm: float = Field(gt=90, lt=230)
    injury_notes: list[str] = Field(default_factory=list)
    baseline_angles: dict[str, float] = Field(default_factory=dict)
    history: dict[str, object] = Field(default_factory=dict)

    @property
    def skill_encoding(self) -> float:
        return {"beginner": 0.0, "intermediate": 0.5, "advanced": 1.0}[self.skill_level]


def load_profile(path: Path) -> UserProfile:
    return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: UserProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")


def update_profile_history(profile: UserProfile, mistakes: list[str]) -> UserProfile:
    recurring = dict(profile.history.get("recurring_mistakes", {}))
    for mistake in mistakes:
        if mistake and mistake != "none":
            recurring[mistake] = int(recurring.get(mistake, 0)) + 1
    history = dict(profile.history)
    history["sessions"] = int(history.get("sessions", 0)) + 1
    history["recurring_mistakes"] = recurring
    return profile.model_copy(update={"history": history})


def load_profiles(directory: Path) -> list[UserProfile]:
    if not directory.exists():
        return []
    profiles: list[UserProfile] = []
    for path in sorted(directory.glob("*.json")):
        profiles.append(load_profile(path))
    return profiles


def profile_to_json(profile: UserProfile) -> str:
    return json.dumps(profile.model_dump(), indent=2, sort_keys=True)
