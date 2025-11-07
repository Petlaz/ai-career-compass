"""Utility helpers for loading datasets and normalizing skill metadata."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from functools import lru_cache
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

_SKILL_SPECIAL_CASES = {
    "ai/ml": "AI/ML",
    "ci/cd": "CI/CD",
    "dbt": "dbt",
    "excel": "Excel",
    "huggingface": "HuggingFace",
    "machine learning": "Machine Learning",
    "mlops": "MLOps",
    "nlp": "NLP",
    "pytorch": "PyTorch",
    "sql": "SQL",
    "aws": "AWS",
    "gcp": "GCP",
    "power bi": "Power BI",
    "bigquery": "BigQuery",
    "open cv": "OpenCV",
    "opencv": "OpenCV",
    "cnn": "CNNs",
    "cnns": "CNNs",
}


def load_roles_csv(path: str | Path = DATA_DIR / "roles_skills_dataset.csv") -> pd.DataFrame:
    """Load the roles dataset and normalize the skill lists."""
    path = Path(path)
    if not path.exists():
        msg = f"Roles dataset not found at {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)

    if "skills_required" not in df.columns:
        msg = "Expected 'skills_required' column in roles dataset."
        logger.error(msg)
        raise ValueError(msg)

    df["skills_required"] = df["skills_required"].apply(parse_skills_column)
    return df


def parse_skills_column(value: str | Iterable[str]) -> List[str]:
    """Parse the skills column that may be stored as a JSON-like string."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            logger.debug("Falling back to comma split parsing for value: %s", value)
            parsed = re.split(r"[,;]\s*", value)
    else:
        parsed = list(value)

    normalized = normalize_skills(parsed)
    return normalized


def normalize_skill(skill: str) -> str:
    """Normalize an individual skill string for consistent comparisons."""
    skill = skill.strip()
    if not skill:
        return skill

    skill_lc = skill.lower()
    if skill_lc in _SKILL_SPECIAL_CASES:
        return _SKILL_SPECIAL_CASES[skill_lc]

    if re.fullmatch(r"[a-z]+(\s[a-z]+)*", skill_lc):
        return skill.title()

    return skill.replace("  ", " ")


def normalize_skills(skills: Iterable[str]) -> List[str]:
    """Normalize a collection of skills and return a sorted, de-duplicated list."""
    normalized = {normalize_skill(skill) for skill in skills if skill and skill.strip()}
    return sorted(normalized)


def load_resources_json(path: str | Path = DATA_DIR / "resources.json") -> list[dict]:
    """Load optional learning resources dataset."""
    path = Path(path)
    if not path.exists():
        logger.info("Resources file not found at %s; returning empty list.", path)
        return []

    with path.open("r", encoding="utf-8") as f:
        resources = json.load(f)
    return resources


@lru_cache(maxsize=1)
def load_skill_vocabulary(path: str | Path = DATA_DIR / "roles_skills_dataset.csv") -> List[str]:
    """Return sorted unique normalized skills from the roles dataset."""
    df = load_roles_csv(path)
    skills: set[str] = set()
    for skill_list in df["skills_required"]:
        skills.update(skill_list)
    return sorted(skills)


__all__ = [
    "DATA_DIR",
    "load_roles_csv",
    "normalize_skills",
    "load_resources_json",
    "parse_skills_column",
    "load_skill_vocabulary",
]
