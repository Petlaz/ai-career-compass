"""Gradio UI for the AI Career Compass recommender."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import gradio as gr
import pandas as pd
from gradio_client import utils as grclient_utils
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommender_engine import (
    RecommenderEngine,
    recommendations_to_list,
    parse_user_skills,
)
from src.resume_parser import extract_skills_from_pdf
from src.explain import STAGES
from src.profiles import list_profiles, get_profile, upsert_profile

RESULT_COLUMNS = [
    "Role",
    "Score",
    "Semantic Similarity",
    "Skill Overlap",
    "Shared Skills",
    "Missing Skills",
    "Explanation",
]

DEFAULT_SUMMARY = "Ready when you are—enter a few skills to discover curated AI/ML career paths."
DEFAULT_RESOURCES = "Missing skills and curated learning resources will appear here after you request recommendations."
DEFAULT_LEARNING_PATH = "A staged learning path (Discover → Build → Apply) will populate after we find missing skills to focus on."

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 48%, #ffffff 100%);
}
#hero {
    background: linear-gradient(120deg, #0f766e, #2563eb);
    color: white;
    border-radius: 28px;
    padding: 40px 48px;
    margin-bottom: 28px;
    box-shadow: 0 22px 45px rgba(15, 23, 42, 0.25);
    position: relative;
    overflow: hidden;
}
#hero::after {
    content: "";
    position: absolute;
    right: -120px;
    top: 20%;
    width: 260px;
    height: 260px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 50%;
    filter: blur(0.5px);
}
#hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
#hero p {
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}
#hero .meta-capsules {
    display: flex;
    gap: 12px;
    margin-top: 20px;
    flex-wrap: wrap;
}
#hero .meta-capsules span {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    padding: 8px 16px;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
#launch-btn {
    display: flex;
    justify-content: center;
    margin-bottom: 8px;
}
#launch-btn button {
    font-size: 1.05rem;
    padding: 0.85rem 1.75rem;
    border-radius: 999px;
    font-weight: 600;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
}
#app-group {
    margin-top: 12px;
}
.tip-card {
    background: white;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 18px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.18);
    margin-bottom: 14px;
}
.tip-card h3 {
    margin-top: 0;
    font-size: 1.05rem;
    color: #0f172a;
}
.tip-card ul {
    padding-left: 18px;
    margin: 12px 0 0;
}
#results-table {
    margin-top: 18px;
}
#results-table tbody tr:hover {
    background: rgba(37, 99, 235, 0.06);
}
#summary-panel, #resource-panel, #learning-path-panel {
    background: white;
    border-radius: 20px;
    padding: 18px 24px;
    box-shadow: 0 18px 26px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.22);
    height: 100%;
}
#resource-panel {
    overflow-y: auto;
}
.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.88rem;
    margin-top: 32px;
}
.gradio-container .tab-nav button {
    border-radius: 999px;
}
.gradio-container .accordion {
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 12px 20px rgba(15, 23, 42, 0.06);
}
"""

logging.basicConfig(level=logging.INFO)

_ORIGINAL_GET_TYPE = grclient_utils.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _ORIGINAL_GET_TYPE(schema)


grclient_utils.get_type = _safe_get_type

engine = RecommenderEngine()
last_recommendation_dicts: List[dict] = []


def _blank_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _default_radar_plot() -> go.Figure:
    return _empty_plot("Skill coverage radar will appear here once you run recommendations.")


def _default_scatter_plot() -> go.Figure:
    return _empty_plot("Semantic vs. skill overlap scatter will render here after scoring.")


def _empty_plot(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=message,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data yet",
                "showarrow": False,
                "font": {"size": 14, "color": "#94a3b8"},
            }
        ],
        height=320,
    )
    return fig


def _normalize_recommendations(recommendations) -> List[dict]:
    if not recommendations:
        return []
    if isinstance(recommendations, list) and recommendations and isinstance(recommendations[0], dict):
        return [dict(rec) for rec in recommendations]
    return recommendations_to_list(list(recommendations))


def _build_visuals(recommendations) -> tuple[go.Figure, go.Figure]:
    normalized = _normalize_recommendations(recommendations)
    if not normalized:
        return _default_radar_plot(), _default_scatter_plot()

    scatter_df = pd.DataFrame(
        [
            {
                "Role": rec.get("role_title", ""),
                "Semantic Similarity": rec.get("semantic_similarity", 0.0),
                "Skill Overlap": rec.get("skill_overlap", 0.0),
                "Hybrid Score": rec.get("score", 0.0),
            }
            for rec in normalized
        ]
    )

    scatter_fig = px.scatter(
        scatter_df,
        x="Semantic Similarity",
        y="Skill Overlap",
        size="Hybrid Score",
        color="Hybrid Score",
        hover_name="Role",
        size_max=26,
        range_x=[0, 1],
        range_y=[0, 1],
        template="plotly_white",
        title="Semantic vs. Skill Overlap",
    )
    scatter_fig.update_layout(height=340, coloraxis_colorbar={"title": "Hybrid"})

    top_recs = normalized[:3]
    categories = ["Hybrid Score", "Semantic Similarity", "Skill Overlap"]
    radar_fig = go.Figure()
    for rec in top_recs:
        values = [
            rec.get("score", 0.0),
            rec.get("semantic_similarity", 0.0),
            rec.get("skill_overlap", 0.0),
        ]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=rec.get("role_title", ""),
            )
        )
    radar_fig.update_layout(
        template="plotly_white",
        title="Top Roles Skill Coverage",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=340,
    )
    return radar_fig, scatter_fig


def _format_learning_path(recommendations) -> str:
    normalized = _normalize_recommendations(recommendations)
    if not normalized:
        return DEFAULT_LEARNING_PATH

    for rec in normalized:
        learning_path = rec.get("learning_path", [])
        if learning_path:
            stages: dict[str, List[str]] = {stage: [] for stage in STAGES}
            for entry in learning_path:
                stage = entry.get("stage", STAGES[0])
                bullet = f"- [{entry.get('title')}]({entry.get('url', '#')}) · {entry.get('type', 'resource').title()} — focus on {', '.join(entry.get('skills', []))}"
                stages.setdefault(stage, []).append(bullet)

            lines = [f"### Learning Path: {rec.get('role_title', '')}"]
            for stage in STAGES:
                entries = stages.get(stage, [])
                if entries:
                    lines.append(f"**{stage}**")
                    lines.extend(entries)
            return "\n".join(lines)
    return DEFAULT_LEARNING_PATH


def _render_outputs(recommendations) -> tuple[pd.DataFrame, str, str, str, go.Figure, go.Figure]:
    normalized = _normalize_recommendations(recommendations)
    global last_recommendation_dicts
    if not normalized:
        last_recommendation_dicts = []
        return _reset_outputs()

    last_recommendation_dicts = normalized
    table_rows = []
    resource_lines: List[str] = []
    for rec in normalized:
        shared_skills = rec.get("shared_skills", [])
        missing_skills = rec.get("missing_skills", [])
        table_rows.append(
            {
                "Role": rec.get("role_title", ""),
                "Score": round(rec.get("score", 0.0), 3),
                "Semantic Similarity": round(rec.get("semantic_similarity", 0.0), 3),
                "Skill Overlap": round(rec.get("skill_overlap", 0.0), 3),
                "Shared Skills": ", ".join(shared_skills) if shared_skills else "-",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "-",
                "Explanation": rec.get("explanation", ""),
            }
        )

        resources = rec.get("resources", []) or []
        if resources:
            resource_lines.append(f"### Next steps for {rec.get('role_title', '')}")
            for resource in resources:
                title = resource.get("title", "Resource")
                url = resource.get("url", "")
                focus = ", ".join(resource.get("skills", []))
                resource_lines.append(f"- 🔗 [{title}]({url}) — strengthen {focus}")

    table = pd.DataFrame(table_rows)[RESULT_COLUMNS]
    top_rec = normalized[0]
    shared = ", ".join(top_rec.get("shared_skills", [])) or "No overlapping skills yet"
    missing = ", ".join(top_rec.get("missing_skills", [])) or "All core skills covered"
    summary_text = (
        f"### Top Match: {top_rec.get('role_title', '')}\n"
        f"- Hybrid score: **{top_rec.get('score', 0.0):.2f}** (70% semantic + 30% skill overlap)\n"
        f"- Shared skills: {shared}\n"
        f"- Focus areas: {missing}"
    )
    resources_markdown = "\n".join(resource_lines) if resource_lines else DEFAULT_RESOURCES
    learning_path_text = _format_learning_path(normalized)
    radar_fig, scatter_fig = _build_visuals(normalized)
    return table, summary_text, resources_markdown, learning_path_text, radar_fig, scatter_fig


def _recommend(user_input: str) -> tuple[pd.DataFrame, str, str, str, go.Figure, go.Figure]:
    recommendations = engine.recommend(user_input, top_k=5)
    recommendation_dicts = recommendations_to_list(recommendations)
    return _render_outputs(recommendation_dicts)


def _reset_outputs() -> tuple[pd.DataFrame, str, str, str, go.Figure, go.Figure]:
    global last_recommendation_dicts
    last_recommendation_dicts = []
    return (
        _blank_results(),
        DEFAULT_SUMMARY,
        DEFAULT_RESOURCES,
        DEFAULT_LEARNING_PATH,
        _default_radar_plot(),
        _default_scatter_plot(),
    )


def _clear_all():
    blank_df, summary, resources, path_text, radar_fig, scatter_fig = _reset_outputs()
    return (
        None,
        "",
        blank_df,
        summary,
        resources,
        path_text,
        radar_fig,
        scatter_fig,
        "",
    )


def _resolve_file_path(file_obj) -> Path | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    # Gradio v4 provides dict-like objects
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    if isinstance(file_obj, dict) and "name" in file_obj:
        return Path(file_obj["name"])
    raise ValueError("Unsupported file object received.")


def _refresh_profiles():
    choices = list_profiles()
    return gr.update(choices=choices)


def _handle_load_profile(profile_name: str):
    if not profile_name:
        blank_df, summary, resources, path_text, radar_fig, scatter_fig = _reset_outputs()
        return (
            "",
            "",
            "",
            blank_df,
            summary,
            resources,
            path_text,
            radar_fig,
            scatter_fig,
            "Select a profile to load.",
        )

    profile = get_profile(profile_name)
    if not profile:
        blank_df, summary, resources, path_text, radar_fig, scatter_fig = _reset_outputs()
        return (
            "",
            "",
            "",
            blank_df,
            summary,
            resources,
            path_text,
            radar_fig,
            scatter_fig,
            f"Profile '{profile_name}' not found.",
        )

    skills = profile.get("skills", [])
    user_input_text = ", ".join(skills)
    notes = profile.get("notes", "") or ""
    recs = profile.get("recommendations") or []
    table, summary_text, resources_text, path_text, radar_fig, scatter_fig = _render_outputs(recs)
    return (
        profile_name,
        notes,
        user_input_text,
        table,
        summary_text,
        resources_text,
        path_text,
        radar_fig,
        scatter_fig,
        f"Loaded profile '{profile_name}'.",
    )


def _handle_save_profile(name: str, notes: str, skills_text: str):
    choices = list_profiles()
    if not name:
        return gr.update(choices=choices), "Please provide a profile name before saving."

    skills = parse_user_skills(skills_text)
    if not skills:
        choices = list_profiles()
        return gr.update(choices=choices, value=name if name in choices else None), "Add at least one skill before saving the profile."

    payload = upsert_profile(
        name=name,
        skills=skills,
        notes=notes or None,
        recommendations=last_recommendation_dicts if last_recommendation_dicts else None,
    )
    choices = list_profiles()
    message = f"Profile '{name}' saved at {payload['saved_at']}."
    return gr.update(choices=choices, value=name), message


def _extract_resume_skills(file_obj, current_text: str) -> tuple[str, pd.DataFrame, str, str, str, go.Figure, go.Figure]:
    file_path = _resolve_file_path(file_obj)
    if not file_path or not file_path.exists():
        blank_df, _, _, path_text, radar_fig, scatter_fig = _reset_outputs()
        return current_text, blank_df, "Upload a PDF resume to extract skills automatically.", DEFAULT_RESOURCES, path_text, radar_fig, scatter_fig

    try:
        skills = extract_skills_from_pdf(file_path)
    except Exception:  # pylint: disable=broad-except
        logging.exception("Failed to parse resume")
        blank_df, _, _, path_text, radar_fig, scatter_fig = _reset_outputs()
        return current_text, blank_df, "Unable to parse resume—please check the file and try again.", DEFAULT_RESOURCES, path_text, radar_fig, scatter_fig

    if not skills:
        blank_df, _, _, path_text, radar_fig, scatter_fig = _reset_outputs()
        return current_text, blank_df, "No matching skills found in the resume. Try adding highlights manually.", DEFAULT_RESOURCES, path_text, radar_fig, scatter_fig

    joined = ", ".join(skills)
    summary = f"Extracted {len(skills)} skills from your resume. Review and edit before requesting recommendations."
    blank_df, _, _, path_text, radar_fig, scatter_fig = _reset_outputs()
    return joined, blank_df, summary, DEFAULT_RESOURCES, path_text, radar_fig, scatter_fig


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="AI Career Compass",
        theme=gr.themes.Soft(primary_hue="teal"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            """
            <section id="hero">
                <h1>AI Career Compass 🧭</h1>
                <p>Transform your current skill stack into curated AI/ML career roles, projects, and learning resources—powered by hybrid semantic search and explainable skill analysis.</p>
                <p style="margin-top: 0.5rem; font-weight: 500;">Built by <strong>Peter Ugonna Obi</strong> — AI/ML Developer</p>
                <div class="meta-capsules">
                    <span>⚙️ Hybrid Semantic Ranking</span>
                    <span>🧠 Sentence-BERT Embeddings</span>
                    <span>🎯 Explainable Skill Gaps</span>
                </div>
            </section>
            """
        )

        launch_btn = gr.Button("Launch App", variant="primary", elem_id="launch-btn")

        with gr.Group(visible=False, elem_id="app-group") as app_group:
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    with gr.Row():
                        profiles_dropdown = gr.Dropdown(
                            choices=list_profiles(),
                            label="Saved profiles",
                            value=None,
                        )
                        refresh_profiles_btn = gr.Button("Refresh", variant="secondary")
                        load_profile_btn = gr.Button("Load Profile", variant="secondary")
                    profile_name_input = gr.Textbox(
                        label="Profile name",
                        placeholder="e.g. ai_career_track",
                    )
                    profile_notes = gr.Textbox(
                        label="Profile notes (optional)",
                        lines=2,
                        placeholder="Add context, goals, or reminders for this profile.",
                    )
                    user_input = gr.Textbox(
                        lines=5,
                        label="Describe your skills, tools, or interests",
                        placeholder="e.g. Python, SQL, TensorFlow, MLOps, Generative AI, healthcare analytics",
                        info="List 4–8 skills, frameworks, or goals to personalize the matches.",
                    )
                    with gr.Row():
                        recommend_btn = gr.Button("Get Recommendations", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                        save_profile_btn = gr.Button("Save Profile", variant="primary")
                    with gr.Row():
                        resume_upload = gr.File(
                            label="Upload resume (PDF)",
                            file_types=[".pdf"],
                            file_count="single",
                        )
                        extract_btn = gr.Button("Extract Skills from Resume", variant="secondary")
                    profile_status = gr.Markdown("", elem_id="profile-status")
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div class="tip-card">
                            <h3>Tips for high-quality matches</h3>
                            <ul>
                                <li>Blend technical skills with domain interests (e.g., "Python, SQL, LLMs, fintech").</li>
                                <li>Add seniority or goals such as "leadership", "portfolio project", or "career switch".</li>
                                <li>Include tools you want to learn to surface stretch opportunities.</li>
                            </ul>
                        </div>
                        """
                    )
                    gr.HTML(
                        """
                        <div class="tip-card">
                            <h3>What you'll receive</h3>
                            <ul>
                                <li>Hybrid score highlighting semantic fit and skill overlap.</li>
                                <li>Shared vs. missing skills for each role.</li>
                                <li>Curated resources mapped to your gap areas.</li>
                            </ul>
                        </div>
                        """
                    )
                    gr.Examples(
                        examples=[
                            ["Python, SQL, TensorFlow, Docker, MLOps"],
                            ["PyTorch, NLP, Transformers, Generative AI, product design"],
                            ["SQL, dbt, Airflow, analytics engineering, stakeholder communication"],
                        ],
                        inputs=user_input,
                        label="Sample skill bundles",
                        examples_per_page=3,
                        run_on_click=False,
                    )

            results = gr.Dataframe(
                value=_blank_results(),
                headers=RESULT_COLUMNS,
                datatype=["str", "number", "number", "number", "str", "str", "str"],
                interactive=False,
                wrap=True,
                elem_id="results-table",
            )

            with gr.Row(equal_height=True):
                summary = gr.Markdown(value=DEFAULT_SUMMARY, elem_id="summary-panel")
                resources = gr.Markdown(value=DEFAULT_RESOURCES, elem_id="resource-panel")
                learning_path = gr.Markdown(value=DEFAULT_LEARNING_PATH, elem_id="learning-path-panel")

            with gr.Row(equal_height=True):
                radar_plot = gr.Plot(value=_default_radar_plot(), label="Skill Coverage Radar")
                scatter_plot = gr.Plot(value=_default_scatter_plot(), label="Semantic vs. Skill Overlap")

            with gr.Accordion("How scoring works", open=False):
                gr.Markdown(
                    "The hybrid score weighs Sentence-BERT semantic similarity (70%) with explicit skill overlap (30%). "
                    "Semantic embeddings capture intent and domain context, while skill matching surfaces shared and missing capabilities for transparent recommendations."
                )

        launch_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=None,
            outputs=[launch_btn, app_group],
        )
        refresh_profiles_btn.click(
            fn=_refresh_profiles,
            inputs=None,
            outputs=profiles_dropdown,
        )
        load_profile_btn.click(
            fn=_handle_load_profile,
            inputs=profiles_dropdown,
            outputs=[
                profile_name_input,
                profile_notes,
                user_input,
                results,
                summary,
                resources,
                learning_path,
                radar_plot,
                scatter_plot,
                profile_status,
            ],
        )
        recommend_btn.click(
            fn=_recommend,
            inputs=user_input,
            outputs=[results, summary, resources, learning_path, radar_plot, scatter_plot],
        )
        save_profile_btn.click(
            fn=_handle_save_profile,
            inputs=[profile_name_input, profile_notes, user_input],
            outputs=[profiles_dropdown, profile_status],
        )
        extract_btn.click(
            fn=_extract_resume_skills,
            inputs=[resume_upload, user_input],
            outputs=[user_input, results, summary, resources, learning_path, radar_plot, scatter_plot],
        )
        clear_btn.click(
            fn=_clear_all,
            inputs=None,
            outputs=[resume_upload, user_input, results, summary, resources, learning_path, radar_plot, scatter_plot, profile_status],
        )

        gr.HTML(
            '<p class="footer-note">Built with Sentence-BERT embeddings, hybrid ranking, and Gradio for rapid AI career exploration.</p>'
        )

    demo.queue(api_open=False)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Career Compass Gradio app.")
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_HOST", "0.0.0.0"),
        help="Host interface to bind (default: 0.0.0.0 for Docker compatibility).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_PORT", "7860")),
        help="Port to serve the app (default: 7860).",
    )
    env_share = os.getenv("GRADIO_SHARE")
    if env_share is None:
        default_share = True
    else:
        default_share = env_share.lower() in {"1", "true", "yes"}

    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument(
        "--share",
        dest="share",
        action="store_true",
        help="Always create a public Gradio share link.",
    )
    share_group.add_argument(
        "--no-share",
        dest="share",
        action="store_false",
        help="Disable creation of a public Gradio share link.",
    )
    parser.set_defaults(share=default_share)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_interface()
    app.launch(server_name=args.host, server_port=args.port, share=args.share, show_api=False)
