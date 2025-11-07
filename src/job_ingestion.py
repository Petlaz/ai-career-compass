"""Utilities for ingesting external job postings into the recommender corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .connectors import JobConnector, JobPosting
from .embeddings import (
    EMBEDDING_CACHE_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    load_embeddings_with_index,
)
from .recommender_engine import _prepare_role_text  # type: ignore
from .utils import DATA_DIR, load_roles_csv, normalize_skills

logger = logging.getLogger(__name__)

JOB_POSTINGS_PATH = DATA_DIR / "job_postings.csv"
MERGED_ROLES_PATH = DATA_DIR / "roles_skills_with_jobs.csv"
SNAPSHOT_PATH = DATA_DIR / "snapshots" / "job_postings.json"


def collect_job_postings(
    connectors: Sequence[JobConnector],
    limit_per_source: int | None = None,
    **filters,
) -> List[JobPosting]:
    """Run the configured connectors and collect postings."""
    postings: List[JobPosting] = []
    for connector in connectors:
        logger.info("Fetching postings from %s", connector.source)
        fetched = list(connector.fetch(**filters))
        if limit_per_source is not None:
            fetched = fetched[:limit_per_source]
        postings.extend(fetched)
    logger.info("Collected %d postings across %d connectors.", len(postings), len(connectors))
    return postings


def postings_to_dataframe(postings: Iterable[JobPosting]) -> pd.DataFrame:
    """Convert job postings into a dataframe ready for persistence."""
    records = []
    for posting in postings:
        record = posting.to_dict()
        record["skills_required"] = normalize_skills(record.get("skills_required", []))
        record["skills_required"] = json.dumps(record["skills_required"])
        records.append(record)
    return pd.DataFrame(records)


def append_postings_to_csv(
    postings: Iterable[JobPosting],
    path: Path = JOB_POSTINGS_PATH,
) -> pd.DataFrame:
    """Append postings to a CSV, deduplicating by source + role + company."""
    df_new = postings_to_dataframe(postings)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df["key"] = (
        df["source"].fillna("") + "|" + df["role_title"].fillna("") + "|" + df["company"].fillna("")
    )
    df = df.drop_duplicates(subset="key").drop(columns=["key"])
    df.to_csv(path, index=False)
    logger.info("Persisted %d job postings to %s", len(df), path)
    return df


def merge_postings_into_roles(
    postings_df: pd.DataFrame,
    base_roles_path: Path = DATA_DIR / "roles_skills_dataset.csv",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Blend job postings into the roles dataset and return the merged dataframe."""
    roles_df = load_roles_csv(base_roles_path)
    postings_roles = postings_df.rename(
        columns={
            "role_title": "role_title",
            "description": "description",
            "skills_required": "skills_required",
        }
    )

    if "industry" not in postings_roles.columns:
        postings_roles["industry"] = "External"

    merged = pd.concat(
        [
            roles_df,
            postings_roles[roles_df.columns.intersection(postings_roles.columns)],
        ],
        ignore_index=True,
    )

    if output_path:
        merged.to_csv(output_path, index=False)
        logger.info("Merged roles dataset written to %s", output_path)

    return merged


def refresh_embeddings(roles_df: pd.DataFrame, force_recompute: bool = True) -> None:
    """Recompute and cache embeddings + FAISS index for the supplied roles dataframe."""
    search_texts = roles_df.apply(_prepare_role_text, axis=1).tolist()
    load_embeddings_with_index(
        search_texts,
        embedding_cache_path=EMBEDDING_CACHE_PATH,
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        force_recompute=force_recompute,
    )
    logger.info("Embedding cache refreshed for %d roles.", len(search_texts))


def export_postings_snapshot(postings: Iterable[JobPosting], path: Path) -> None:
    """Write job postings to JSON for audit/debug purposes."""
    payload = [posting.to_dict() for posting in postings]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Snapshot written to %s", path)


def run_sample_harvest(force_recompute_embeddings: bool = True) -> dict:
    """End-to-end sample pipeline using the local connector datasets."""
    from .connectors import KaggleConnector, LinkedInConnector

    connectors: List[JobConnector] = [
        LinkedInConnector(),
        KaggleConnector(),
    ]

    postings = collect_job_postings(connectors)
    if not postings:
        logger.warning("No postings collected; aborting ingestion.")
        return {"postings_count": 0, "merged_roles": 0}

    export_postings_snapshot(postings, SNAPSHOT_PATH)
    postings_df = append_postings_to_csv(postings, path=JOB_POSTINGS_PATH)
    merged_roles = merge_postings_into_roles(postings_df, output_path=MERGED_ROLES_PATH)

    if force_recompute_embeddings:
        refresh_embeddings(merged_roles, force_recompute=True)

    summary = {
        "postings_count": len(postings_df),
        "merged_roles": len(merged_roles),
        "snapshot": str(SNAPSHOT_PATH),
        "postings_csv": str(JOB_POSTINGS_PATH),
        "merged_roles_csv": str(MERGED_ROLES_PATH),
    }
    logger.info("Sample harvest summary: %s", summary)
    return summary
