---
title: AI Career Compass 🧭
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app/gradio_app.py
pinned: false
license: mit
---

# 🧭 AI Career Compass  
**Discover your next AI/ML career path.**

This Gradio app uses semantic search with Sentence-BERT embeddings to match your skills to relevant AI/ML roles and learning resources.

## 🚀 How to Use
1. Enter your skills (e.g. “Python, TensorFlow, SQL”).  
2. Click **Get Recommendations**.  
3. View your top matches and missing skills.

Built with 💙 by **Peter Ugonna Obi** — deployed on Hugging Face Spaces.


# AI Career Compass – Roadmap & Checklist

## ✅ Completed Foundations

- **Baseline Dataset & Utilities**
  - `data/roles_skills_dataset.csv` seeded with AI/ML roles.
  - `src/utils.py` normalizes skills and loads datasets.
- **Sentence-BERT Embeddings**
  - `src/embeddings.py` loads `all-MiniLM-L6-v2`, caches vectors, and builds a FAISS index.
- **Hybrid Recommender Engine**
  - `src/recommender_engine.py` blends semantic similarity (FAISS) with skill overlap.
  - Integrates explainability from `src/explain.py`.
- **Explainable Gradio UI**
  - `app/gradio_app.py` provides polished UX, summary panels, resource suggestions, and resume upload.
- **Plotly Visual Analytics**
  - Radar and scatter charts highlight semantic vs. skill overlap directly in the Gradio UI.
- **Learning Path Generator**
  - Discover → Build → Apply pathways derived from missing-skill resources surfaced in the Gradio app.
- **User Profiles & Persistence**
  - `src/profiles.py` stores normalized skill snapshots and recommendation history in `data/profiles.json`.
- **Resume Skill Extraction**
  - `src/resume_parser.py` parses PDFs via `pdfplumber` and feeds normalized skills into the pipeline.
- **Docker + Docs**
  - Dockerfile for container builds, professional README, requirements pinned for macOS CPU.

## 🚧 In Progress

- **Live Job API Integrations**
  - ✅ Connector scaffolding (`src/connectors/`) and ingestion helpers (`src/job_ingestion.py`, `data/job_postings.csv`).
  - 🛠️ Ship OAuth prototypes for LinkedIn and Kaggle connectors; validate token refresh flow.
    - Owner: Integrations pod (lead: Peter)
    - Target: 2024-07-12
  - 🛠️ Stand up ingestion job to persist roles to CSV/DB and trigger embedding re-cache via CI.
    - Owner: Data & Platform pod
    - Target: 2024-07-26
  - 🛠️ Schedule nightly backfill in Airflow/Cron and capture monitoring hooks for failures.
    - Owner: Platform & Ops pod
    - Target: 2024-08-02

## ⏭ Up Next
- **Deployment to Hugging Face Space**
  - Stand up a public Space for the Gradio app with automated builds & telemetry.

## 📌 Future Enhancements

- **Resume Parsing Enhancements**
  - Integrate PyMuPDF / spaCy NER for richer extraction.
  - Support plain text and DOCX ingestion.
  - Wire CI/CD to push Space updates from the main branch.

Use this checklist to prioritize upcoming sprints and keep the AI Career Compass roadmap transparent. Contributions welcome!
