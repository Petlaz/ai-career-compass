"""Connector interfaces for external job data sources."""

from .base import JobPosting, JobConnector
from .kaggle import KaggleConnector
from .linkedin import LinkedInConnector

__all__ = [
    "JobPosting",
    "JobConnector",
    "KaggleConnector",
    "LinkedInConnector",
]
