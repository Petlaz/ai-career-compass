"""Placeholder connector for Kaggle datasets or competitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from .base import JobConnector, JobPosting

logger = logging.getLogger(__name__)


class KaggleConnector(JobConnector):
    """Stub connector for ingesting Kaggle project ideas or job listings."""

    source = "kaggle"

    def __init__(
        self,
        username: str | None = None,
        token: str | None = None,
        dataset_path: str | Path | None = Path(__file__).resolve().parents[2] / "data" / "kaggle_projects_sample.json",
    ) -> None:
        self.username = username
        self.token = token
        self.dataset_path = Path(dataset_path) if dataset_path else None

    def fetch(self, **kwargs) -> Iterable[JobPosting]:
        """Yield project or job opportunities from Kaggle.

        Replace this stub with calls to the Kaggle REST API or local datasets.
        """
        logger.info("Kaggle connector called with filters: %s", kwargs)
        if self.dataset_path and self.dataset_path.exists():
            return self._load_from_file()
        if not (self.username and self.token):
            logger.warning("Kaggle dataset not found and credentials missing; returning no postings.")
            return []

        # Placeholder for future implementation
        return []

    def _load_from_file(self) -> List[JobPosting]:
        import json

        with self.dataset_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        postings = []
        for record in records:
            postings.append(
                JobPosting(
                    source=self.source,
                    role_title=record["role_title"],
                    description=record["description"],
                    skills_required=self.normalize_skills(record.get("skills_required", [])),
                    company=record.get("company"),
                    location=record.get("location"),
                    url=record.get("url"),
                    extras=record.get("extras"),
                )
            )
        logger.info("Loaded %d Kaggle sample postings from %s", len(postings), self.dataset_path)
        return postings
