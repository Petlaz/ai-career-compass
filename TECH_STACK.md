# Tech Stack — AI Career Compass

## Language & Runtime
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Primary programming language |

---

## Machine Learning & AI
| Technology | Version | Purpose |
|---|---|---|
| PyTorch | 2.3.0 | Deep learning framework (tensor ops, model backend) |
| torchvision | 0.18.0 | PyTorch image utilities |
| torchaudio | 2.3.0 | PyTorch audio utilities |
| sentence-transformers | 2.6.0 | Sentence-BERT embeddings (`all-MiniLM-L6-v2` model) |
| scikit-learn | >=1.3,<2.0 | ML utilities (scoring, similarity helpers) |

---

## Vector Search & Embeddings
| Technology | Version | Purpose |
|---|---|---|
| FAISS (faiss-cpu) | 1.7.4 | Fast approximate nearest-neighbour vector index |
| ChromaDB | 0.4.24 | Vector database for embedding persistence |

---

## Data Processing
| Technology | Version | Purpose |
|---|---|---|
| NumPy | 1.26.4 | Numerical array operations & embedding normalization |
| Pandas | >=2.1,<3.0 | Tabular data handling (roles CSV, job postings CSV) |

---

## User Interface
| Technology | Version | Purpose |
|---|---|---|
| Gradio | 4.44.0 | Interactive web UI for the career recommender |
| Plotly | >=5.18,<6.0 | Interactive charts & visualizations in the UI |

---

## Document Parsing
| Technology | Version | Purpose |
|---|---|---|
| pdfplumber | >=0.11,<1.0 | PDF resume parsing & skill extraction |

---

## Utilities
| Technology | Version | Purpose |
|---|---|---|
| tqdm | >=4.66 | Progress bars for long-running operations |

---

## Data Sources & Formats
| Format / Source | Purpose |
|---|---|
| CSV (`roles_skills_dataset.csv`, `job_postings.csv`) | Role definitions and job postings corpus |
| JSON (`profiles.json`, `resources.json`, `kaggle_projects_sample.json`, `linkedin_jobs_sample.json`) | User profiles, learning resources, external job data |
| PDF (resume uploads) | User resume ingestion via `pdfplumber` |

---

## External Connectors (Stub / Extensible)
| Connector | Purpose |
|---|---|
| LinkedIn connector (`src/connectors/linkedin.py`) | Fetches LinkedIn job postings |
| Kaggle connector (`src/connectors/kaggle.py`) | Fetches Kaggle datasets / project listings |

---

## Infrastructure & Deployment
| Technology | Purpose |
|---|---|
| Docker | Containerised deployment via `Dockerfile` (base image: `python:3.10-slim`, port `7860`) |

---

## Architecture Overview

```
User (Browser)
    │
    ▼
Gradio UI  (app/gradio_app.py)
    │
    ├── Resume Parser  ──► pdfplumber
    │
    ├── Recommender Engine  (src/recommender_engine.py)
    │       ├── Semantic Search  ──► Sentence-BERT + FAISS
    │       └── Skill Overlap Scoring  ──► NumPy / set operations
    │
    ├── Explainability  (src/explain.py)
    │       └── Skill gap analysis + learning path generation
    │
    ├── Job Ingestion  (src/job_ingestion.py)
    │       └── Connectors: LinkedIn, Kaggle
    │
    └── Profiles  (src/profiles.py)
            └── JSON-backed user profile store
```
