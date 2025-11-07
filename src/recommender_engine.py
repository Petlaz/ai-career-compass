"""Hybrid semantic recommender for AI career roles."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .embeddings import (
    EMBEDDING_CACHE_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    load_embeddings_with_index,
)
from .explain import generate_explanation, suggest_resources, generate_learning_path
from .utils import load_resources_json, load_roles_csv, normalize_skills

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.7
SKILL_WEIGHT = 0.3


def _prepare_role_text(role_row) -> str:
    skills = ", ".join(role_row["skills_required"])
    components = [
        role_row.get("role_title", ""),
        role_row.get("description", ""),
        f"Skills: {skills}",
    ]
    return ". ".join(component for component in components if component)


def parse_user_skills(raw_input: str) -> List[str]:
    """Parse the user's free-form skill input into a normalized list."""
    if not raw_input:
        return []
    tokens = re.split(r"[,\n;/]+", raw_input)
    return normalize_skills(token for token in tokens if token.strip())


@dataclass
class Recommendation:
    role_title: str
    score: float
    semantic_similarity: float
    skill_overlap: float
    explanation: str
    shared_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    resources: List[dict] = field(default_factory=list)
    learning_path: List[dict] = field(default_factory=list)


class RecommenderEngine:
    """Coordinate semantic and skill-based ranking for career roles."""

    def __init__(
        self,
        roles_df=None,
        resources: Sequence[dict] | None = None,
        cache_path=EMBEDDING_CACHE_PATH,
    ) -> None:
        self.roles_df = roles_df if roles_df is not None else load_roles_csv()
        self.resources = list(resources) if resources is not None else load_resources_json()
        self.cache_path = cache_path

        self.roles_df = self.roles_df.copy()
        self.roles_df["search_text"] = self.roles_df.apply(_prepare_role_text, axis=1)
        self.role_embeddings, self.faiss_index = load_embeddings_with_index(
            self.roles_df["search_text"].tolist(),
            embedding_cache_path=self.cache_path,
            index_path=FAISS_INDEX_PATH,
            metadata_path=FAISS_METADATA_PATH,
        )

    def recommend(self, user_input: str, top_k: int = 5) -> List[Recommendation]:
        if self.roles_df.empty:
            logger.warning("Roles dataset is empty; returning no recommendations.")
            return []

        user_skills = parse_user_skills(user_input)
        user_text = user_input.strip() or " ".join(user_skills)
        if not user_text:
            logger.info("No user input provided; returning empty recommendations.")
            return []

        from .embeddings import get_embeddings  # Local import avoids circular import at top

        user_vector = get_embeddings([user_text])[0]
        user_vector32 = user_vector.astype("float32", copy=False).reshape(1, -1)
        distances, indices = self.faiss_index.search(user_vector32, len(self.roles_df))
        semantic_scores = np.zeros(len(self.roles_df), dtype=float)
        semantic_scores[indices[0]] = distances[0]

        recommendations: List[Recommendation] = []
        for idx, role_row in self.roles_df.iterrows():
            analysis = generate_explanation(user_skills, role_row["skills_required"])
            skill_overlap = analysis["coverage"]
            final_score = SEMANTIC_WEIGHT * semantic_scores[idx] + SKILL_WEIGHT * skill_overlap
            rec = Recommendation(
                role_title=role_row.get("role_title", ""),
                score=float(final_score),
                semantic_similarity=float(semantic_scores[idx]),
                skill_overlap=float(skill_overlap),
                explanation=analysis["explanation"],
                shared_skills=analysis["shared"],
                missing_skills=analysis["missing"],
            )
            rec.resources = suggest_resources(self.resources, rec.missing_skills)
            rec.learning_path = generate_learning_path(self.resources, rec.missing_skills)
            recommendations.append(rec)

        recommendations.sort(key=lambda rec: rec.score, reverse=True)
        return recommendations[:top_k]


def recommendation_to_dict(rec: Recommendation) -> dict:
    return {
        "role_title": rec.role_title,
        "score": rec.score,
        "semantic_similarity": rec.semantic_similarity,
        "skill_overlap": rec.skill_overlap,
        "explanation": rec.explanation,
        "shared_skills": rec.shared_skills,
        "missing_skills": rec.missing_skills,
        "resources": rec.resources,
        "learning_path": rec.learning_path,
    }


def recommendations_to_list(recs: List[Recommendation]) -> List[dict]:
    return [recommendation_to_dict(rec) for rec in recs]


__all__ = [
    "RecommenderEngine",
    "Recommendation",
    "parse_user_skills",
    "recommendation_to_dict",
    "recommendations_to_list",
]
