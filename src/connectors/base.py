"""Base interfaces for live job data connectors."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional


@dataclass
class JobPosting:
    """Lightweight container for normalized job metadata."""

    source: str
    role_title: str
    description: str
    skills_required: List[str] = field(default_factory=list)
    company: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    extras: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        payload = {
            "source": self.source,
            "role_title": self.role_title,
            "description": self.description,
            "skills_required": self.skills_required,
            "company": self.company,
            "location": self.location,
            "url": self.url,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


class JobConnector(abc.ABC):
    """Abstract connector definition for external job sources."""

    source: str

    @abc.abstractmethod
    def fetch(self, **kwargs) -> Iterable[JobPosting]:
        """Yield job postings from the external system."""

    @staticmethod
    def normalize_skills(skills: Iterable[str]) -> List[str]:
        from src.utils import normalize_skills  # lazy import to avoid cycles

        return normalize_skills(skills)
