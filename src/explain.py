"""Explainability utilities for surfacing shared and missing skills."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def compare_skills(user_skills: Sequence[str], target_skills: Sequence[str]) -> dict:
    """Return shared and missing skills plus coverage ratio."""
    user_set = {skill for skill in user_skills}
    target_set = {skill for skill in target_skills}

    shared = sorted(user_set & target_set)
    missing = sorted(target_set - user_set)
    coverage = len(shared) / len(target_set) if target_set else 0.0

    return {
        "shared": shared,
        "missing": missing,
        "coverage": coverage,
    }


def format_explanation(shared: Iterable[str], missing: Iterable[str], coverage: float) -> str:
    """Create a human-readable explanation string."""
    coverage_pct = int(round(coverage * 100))
    shared_str = ", ".join(shared) if shared else "none of the listed skills"
    if missing:
        missing_str = ", ".join(missing)
        return f"Matched {coverage_pct}% of your skills: {shared_str}; missing: {missing_str}."
    return f"Matched {coverage_pct}% of your skills: {shared_str}."


def suggest_resources(resources: List[dict], missing_skills: Sequence[str], limit: int = 3) -> List[dict]:
    """Suggest follow-up resources based on missing skills."""
    if not resources or not missing_skills:
        return []

    missing_set = {skill.lower() for skill in missing_skills}
    scored = []
    for resource in resources:
        resource_skills = {skill.lower() for skill in resource.get("skills", [])}
        overlap = missing_set & resource_skills
        if overlap:
            scored.append((len(overlap), resource))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


STAGES = ["Discover", "Build", "Apply"]


def generate_learning_path(resources: List[dict], missing_skills: Sequence[str]) -> List[dict]:
    """Construct a staged learning path based on missing skills."""
    if not resources or not missing_skills:
        return []

    staged_resources = suggest_resources(resources, missing_skills, limit=9)
    if not staged_resources:
        return []

    path: List[dict] = []
    stage_index = 0
    for idx, resource in enumerate(staged_resources):
        stage = STAGES[min(stage_index, len(STAGES) - 1)]
        path.append(
            {
                "stage": stage,
                "title": resource.get("title"),
                "url": resource.get("url"),
                "skills": resource.get("skills", []),
                "type": resource.get("type", "resource"),
            }
        )
        if (idx + 1) % 3 == 0:
            stage_index += 1
    return path


def generate_explanation(user_skills: Sequence[str], target_skills: Sequence[str]) -> dict:
    """High-level helper returning explanation data and formatted string."""
    analysis = compare_skills(user_skills, target_skills)
    analysis["explanation"] = format_explanation(
        analysis["shared"],
        analysis["missing"],
        analysis["coverage"],
    )
    return analysis


__all__ = [
    "compare_skills",
    "format_explanation",
    "generate_explanation",
    "suggest_resources",
    "generate_learning_path",
    "STAGES",
]
