---
license: mit
title: AI Career Compass 🧭
sdk: gradio
emoji: ⚡
app_file: app/gradio_app.py
colorFrom: indigo
colorTo: blue
pinned: false
short_description: Build AI Career Compass
---


# 🧭 AI Career Compass

![AI Career Compass Demo](space_thumbnail.png)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face%20Space-AI%20Career%20Compass-blue)](https://huggingface.co/spaces/petlaz/ai-career-compass)
[![Gradio App](https://img.shields.io/badge/Powered%20by-Gradio-orange)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Deploy to HF Space](https://img.shields.io/badge/Deploy-Hugging%20Face%20Space-yellowgreen)](https://huggingface.co/spaces/create)

Skill-to-Opportunity Recommender & Semantic Search Engine

---

## 🔗 Live Demo
- Hugging Face Space: https://huggingface.co/spaces/petlaz/ai-career-compass

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Setup & Installation](#️-setup--installation)
6. [Docker Deployment](#-docker-deployment)
7. [Live Job Ingestion](#-live-job-ingestion-scaffolding)
8. [Learning Path Generator](#-learning-path-generator)
9. [User Profiles](#-user-profiles-persistence)
10. [Example Output](#-example-output)
11. [Roadmap](#-roadmap--checklist)
12. [Author](#-author)
13. [License](#-license)
14. [Clone & Run Locally](#-clone--run-locally)

---

## 🧩 Overview

AI Career Compass is a personalized AI/ML career recommender system and semantic search engine that intelligently matches a user’s skills and interests with relevant roles, projects, and learning opportunities.

Built on transformer-based embeddings and hybrid similarity scoring, the platform delivers explainable recommendations so learners can identify their best-fit roles and the skills to develop next.

> 💡 Think of it as an AI mentor for your career journey.

> 👤 **Gradio App Builder:** Peter Ugonna Obi — AI/ML Developer

---

## 🎯 Key Features

- **🧠 Semantic Understanding** – Sentence-BERT (`all-MiniLM-L6-v2`) encodes user inputs and role descriptions for context-aware matching.
- **🔍 Hybrid Recommender Engine** – Combines cosine similarity with explicit skill-overlap scoring (70/30 weighting).
- **💬 Explainability** – Surfaces shared and missing skills for each recommendation, plus curated resources for gaps.
- **💻 Interactive Gradio Demo** – Beautiful, responsive UI for capturing skills/goals and exploring recommendations.
- **⚡ Vector Indexing** – FAISS-powered inner-product search over normalized embeddings for fast retrieval.
- **📁 Resume Skill Extraction** – Upload a PDF resume to auto-populate the skill prompt.
- **📊 Visual Insights** – Plotly radar and scatter charts to compare hybrid scores, semantic match, and skill coverage.
- **🧭 Learning Path Generator** – Stage recommended resources into Discover → Build → Apply pathways.
- **🐳 Dockerized Deployment** – Build-and-run container for a consistent, portable runtime.
- **🍎 Mac-Friendly** – CPU-only stack (no CUDA) for seamless macOS development.

---

## 🛠 Tech Stack

- **Language & Runtime:** Python 3.11, virtualenv
- **Modeling:** SentenceTransformers, PyTorch (CPU wheels), FAISS
- **Data Layer:** pandas, NumPy, pdfplumber, custom CSV/JSON datasets
- **Application:** Gradio 4.x, Plotly, Hugging Face Spaces
- **Ops & Packaging:** Docker, huggingface_hub CLI

---

## 🧱 Project Structure

```
ai_career_compass/
├── app/
│   └── gradio_app.py          # Enhanced Gradio interface
├── data/
│   ├── job_postings.csv
│   ├── roles_skills_dataset.csv
│   ├── resources.json
│   ├── linkedin_jobs_sample.json
│   ├── kaggle_projects_sample.json
│   └── profiles.json
├── src/
│   ├── connectors/            # Live job API connector scaffolding
│   ├── embeddings.py          # Sentence-BERT loading & caching
│   ├── explain.py             # Skill comparison and narratives
│   ├── recommender_engine.py  # Hybrid ranking logic
│   ├── job_ingestion.py       # Merge external job data & refresh embeddings
│   ├── resume_parser.py       # Resume PDF skill extraction helper
│   ├── profiles.py            # User profile persistence helpers
│   └── utils.py               # Data loading & normalization helpers
├── requirements.txt
├── Dockerfile
├── ROADMAP.md
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone & Create a Virtual Environment

```bash
git clone https://huggingface.co/spaces/petlaz/ai-career-compass.git
cd ai_career_compass
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies (macOS CPU-Friendly)

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Run the App Locally

```bash
python app/gradio_app.py
```

- Visit **http://127.0.0.1:7860**
- On the landing screen, click **Launch App**
- Optional: Upload a PDF resume to auto-fill skills via the **Extract Skills from Resume** button.
- By default the server shares both a `Local URL` and a `Public URL`. Disable sharing with `--no-share` if desired.

To explicitly request a public Gradio link:

```bash
python app/gradio_app.py --share
```

---

## 🚀 Docker Deployment

Build and run the containerized app:

```bash
docker build -t ai-career-compass .
docker run --rm -p 7860:7860 ai-career-compass
```

- Navigate to **http://127.0.0.1:7860**
- Click **Launch App** to begin
- Pass `--no-share` to disable public Gradio URLs inside Docker:

```bash
docker run --rm -p 7860:7860 ai-career-compass python app/gradio_app.py --no-share
```

Otherwise, watch the container logs for the generated `Public URL`. Resume uploads work the same inside the container.

---

## 🧮 How It Works

1. **User Input** – Capture skills, tools, and interests via Gradio.
2. **Embeddings** – Encode role descriptions and user input with Sentence-BERT and cache them.
3. **Resume Parsing (optional)** – Extract skills from uploaded PDFs and merge with manual inputs.
4. **Semantic Search** – Query a FAISS inner-product index to rank roles by semantic fit.
5. **Hybrid Ranking** – Combine semantic and skill-overlap scores (70% semantic, 30% overlap).
6. **Explainability** – Highlight shared vs. missing skills and recommend learning resources.
7. **Visualization** – Render Plotly radar/scatter charts to contextualize semantic vs. skill overlap.
8. **Learning Path Generator** – Organize missing-skill resources into staged Discover → Build → Apply journeys.
9. **Output** – Display top-ranked roles with hybrid scores and actionable insights.

---

## 🌐 Live Job Ingestion (Scaffolding)

`src/connectors/` defines pluggable connectors (`LinkedInConnector`, `KaggleConnector`) ready to be wired to real APIs once credentials are available. Use `src/job_ingestion.py` to:

1. Collect postings from connectors.
2. Append them to `data/job_postings.csv` (deduplicated).
3. Merge them into the roles corpus and trigger an embedding refresh.

Example pseudo-workflow:

```python
from src.connectors import LinkedInConnector, KaggleConnector
from src.job_ingestion import collect_job_postings, append_postings_to_csv, merge_postings_into_roles, refresh_embeddings

connectors = [
    LinkedInConnector(api_token="..."),
    KaggleConnector(username="...", token="..."),
]

postings = collect_job_postings(connectors, limit_per_source=100, keywords=["Generative AI"])
df_postings = append_postings_to_csv(postings)
merged_roles = merge_postings_into_roles(df_postings)
refresh_embeddings(merged_roles)
```

All connectors return `JobPosting` objects, making it easy to add new sources without changing downstream code.

For a quick local demo, call:

```bash
python -c "from src.job_ingestion import run_sample_harvest; print(run_sample_harvest())"
```

This uses the bundled sample datasets (`data/linkedin_jobs_sample.json`, `data/kaggle_projects_sample.json`) to populate `data/job_postings.csv`, merge into `data/roles_skills_with_jobs.csv`, and refresh embeddings.

---

## 🧭 Learning Path Generator

Each recommendation surfaces a staged journey:

1. **Discover** – foundational reading or courses to understand concepts.
2. **Build** – intermediate projects or labs to practice skills.
3. **Apply** – advanced builds or portfolio proofs to showcase mastery.

The panel updates dynamically based on missing skills, drawing from `data/resources.json`. Add more resources keyed by skill keywords to expand the pathway suggestions.

---

## 💾 User Profiles (Persistence)

Use `src/profiles.py` to save and reload personalized skill snapshots and recommendation history:

```python
from src.recommender_engine import RecommenderEngine, recommendations_to_list
from src.profiles import upsert_profile, list_profiles

engine = RecommenderEngine()
recs = engine.recommend("Python, TensorFlow, ML Ops, AWS")

upsert_profile(
    name="peter",
    skills=["Python", "TensorFlow", "MLOps", "AWS"],
    recommendations=recommendations_to_list(recs),
    notes="Targeting ML platform leadership roles.",
)

print(list_profiles())
```

Profiles are stored in `data/profiles.json`; each entry tracks normalized skills, optional notes, timestamps, and the latest recommendation snapshot. Future UI work can surface profile pickers or saved progress bars.

---

## 🧠 Example Output

**Input**

```
Python, Pandas, machine learning, deep learning
```

**Output**

| Rank | Recommended Role            | Match Score | Key Insight                                         |
|------|-----------------------------|-------------|-----------------------------------------------------|
| 1    | 🧠 Machine Learning Engineer | 0.87        | Matches 80% of your skills; missing: Docker, CI/CD  |
| 2    | 📊 Data Scientist            | 0.84        | Strong match; missing: Power BI, SQL                |
| 3    | 🤖 AI Developer              | 0.81        | Good overlap; add TensorFlow + FastAPI              |

---

## 🧭 Roadmap & Checklist

- ✅ **Semantic + Hybrid Core** – Sentence-BERT embeddings with FAISS-backed ranking and explainable skill overlap.
- ✅ **Resume Skill Extraction** – Upload a PDF to auto-fill normalized skills via `src/resume_parser.py`.
- 🛠️ **Live Job APIs** – Build connectors (LinkedIn, Kaggle, etc.) to ingest fresh roles, persist skills, and trigger re-embedding.
- 📊 **Plotly Visuals** – Surface radar/scatter charts in Gradio to visualize coverage and semantic fit.
- 🧭 **Learning Path Generator** – Recommend staged resources/projects for missing skills.

---

## 🧑‍💻 Author

**Peter Ugonna Obi**

- 🎓 M.Sc. Communication Systems & Networks – TH Köln  
- 💼 Aspiring Machine Learning Engineer / AI Developer  
- 📍 Cologne, Germany  
- 🔗 [LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/)

---

## ⚖️ License

MIT License © 2025 – Peter Ugonna Obi

---

## 🧩 Clone & Run Locally

You can clone and run this project locally on macOS or any CPU-based environment.

```bash
git clone https://huggingface.co/spaces/petlaz/ai-career-compass.git
cd ai-career-compass
pip install -r requirements.txt
python app/gradio_app.py
```

The Gradio app will start locally at 👉 http://127.0.0.1:7860
