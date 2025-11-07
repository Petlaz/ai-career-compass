"""Utilities for extracting skills from resume PDFs."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import pdfplumber

from .utils import load_skill_vocabulary, normalize_skills

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract raw text from a PDF resume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found at {path}")

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    text = "\n".join(pages)
    logger.debug("Extracted %d characters from %s", len(text), path)
    return text


def _build_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill.lower())
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=None)
def _compiled_patterns(skills: Sequence[str]) -> List[tuple[str, re.Pattern[str]]]:
    return [(skill, _build_pattern(skill)) for skill in skills]


def extract_skills_from_text(
    text: str,
    skill_vocabulary: Sequence[str],
    extra_keywords: Iterable[str] | None = None,
) -> List[str]:
    """Extract normalized skills from raw resume text."""
    text_lower = text.lower()
    matches = set()

    for skill, pattern in _compiled_patterns(tuple(skill_vocabulary)):
        if pattern.search(text_lower):
            matches.add(skill)

    if extra_keywords:
        normalized_extras = normalize_skills(extra_keywords)
        for keyword in normalized_extras:
            if keyword.lower() in text_lower:
                matches.add(keyword)

    extracted = normalize_skills(matches)
    logger.debug("Extracted %d skills from resume text.", len(extracted))
    return extracted


def extract_skills_from_pdf(
    path: str | Path,
    skill_vocabulary: Sequence[str] | None = None,
    extra_keywords: Iterable[str] | None = None,
) -> List[str]:
    """Extract a list of normalized skills from a resume PDF."""
    if skill_vocabulary is None:
        skill_vocabulary = load_skill_vocabulary()

    text = extract_text_from_pdf(path)
    return extract_skills_from_text(text, skill_vocabulary, extra_keywords=extra_keywords)


__all__ = [
    "extract_text_from_pdf",
    "extract_skills_from_pdf",
    "extract_skills_from_text",
]
