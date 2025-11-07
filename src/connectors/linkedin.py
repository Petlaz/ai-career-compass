"""Placeholder connector for LinkedIn job listings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from .base import JobConnector, JobPosting

logger = logging.getLogger(__name__)


class LinkedInConnector(JobConnector):
    """Stub connector demonstrating the contract for LinkedIn ingestion."""

    source = "linkedin"

    def __init__(
        self,
        api_token: str | None = None,
        dataset_path: str | Path | None = Path(__file__).resolve().parents[2] / "data" / "linkedin_jobs_sample.json",
    ) -> None:
        self.api_token = api_token
        self.dataset_path = Path(dataset_path) if dataset_path else None

    def fetch(self, **kwargs) -> Iterable[JobPosting]:
        """Yield job postings from LinkedIn.

        This stub produces no data. Replace with real API/HTML scraping logic
        once credentials are available.
        """
        logger.info("LinkedIn connector called with filters: %s", kwargs)
        if self.dataset_path and self.dataset_path.exists():
            return self._load_from_file()
        if not self.api_token:
            logger.warning("LinkedIn dataset not found and API token missing; returning no postings.")
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
        logger.info("Loaded %d LinkedIn sample postings from %s", len(postings), self.dataset_path)
        return postings
