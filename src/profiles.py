"""User profile persistence utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .utils import DATA_DIR, normalize_skills

logger = logging.getLogger(__name__)

PROFILES_PATH = DATA_DIR / "profiles.json"


@dataclass
class UserProfile:
    name: str
    skills: List[str]
    notes: Optional[str] = None
    saved_at: str = datetime.now(timezone.utc).isoformat()
    recommendations: List[dict] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["skills"] = sorted(set(normalize_skills(self.skills)))
        return payload


def _read_profiles(path: Path = PROFILES_PATH) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        profiles = {item["name"]: item for item in data if "name" in item}
    else:
        profiles = {name: payload for name, payload in data.items()}
    return profiles


def _write_profiles(profiles: Dict[str, dict], path: Path = PROFILES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = list(profiles.values())
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    logger.info("Persisted %d profiles to %s", len(serialized), path)


def list_profiles(path: Path = PROFILES_PATH) -> List[str]:
    profiles = _read_profiles(path)
    return sorted(profiles.keys())


def get_profile(name: str, path: Path = PROFILES_PATH) -> dict | None:
    profiles = _read_profiles(path)
    return profiles.get(name)


def upsert_profile(
    name: str,
    skills: Iterable[str],
    notes: str | None = None,
    recommendations: Iterable[dict] | None = None,
    path: Path = PROFILES_PATH,
) -> dict:
    profiles = _read_profiles(path)
    profile = UserProfile(
        name=name,
        skills=list(skills),
        notes=notes,
        recommendations=list(recommendations) if recommendations else None,
        saved_at=datetime.now(timezone.utc).isoformat(),
    )
    payload = profile.to_dict()
    profiles[name] = payload
    _write_profiles(profiles, path)
    return payload


def delete_profile(name: str, path: Path = PROFILES_PATH) -> bool:
    profiles = _read_profiles(path)
    if name not in profiles:
        return False
    profiles.pop(name)
    _write_profiles(profiles, path)
    return True


__all__ = [
    "UserProfile",
    "PROFILES_PATH",
    "list_profiles",
    "get_profile",
    "upsert_profile",
    "delete_profile",
]
