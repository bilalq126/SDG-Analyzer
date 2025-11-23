# utils.py
"""
Utility functions for EcoMind AI Streamlit app.
- create_radar_chart: uses plotly to render a radar/spider chart for SDG scores
- sdg_color_map / sdg_name_map: small helpers for display
"""

from typing import Dict, List, Tuple
import plotly.graph_objects as go
import math

# Minimal SDG names mapping (1-17). You can expand these if desired.
SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water",
    7: "Affordable Energy",
    8: "Decent Work",
    9: "Industry/Innovation",
    10: "Reduced Inequalities",
    11: "Sustainable Cities",
    12: "Responsible Consumption",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace & Justice",
    17: "Partnerships"
}

def get_sdg_name(sdg_id: int) -> str:
    return SDG_NAMES.get(sdg_id, f"SDG {sdg_id}")

def create_radar_chart(sdgs: List[Dict], title: str = "SDG Relevance") -> go.Figure:
    """
    sdgs: list of dicts with keys 'id', 'short_name' (optional), 'score' (0-100)
    Returns a Plotly Figure (radar/spider chart).
    """
    # Prepare labels and values
    labels = []
    values = []
    for s in sdgs:
        sid = s.get("id")
        name = s.get("short_name") or get_sdg_name(sid)
        labels.append(f"{sid}: {name}")
        # Ensure numeric
        try:
            score = float(s.get("score", 0))
        except Exception:
            score = 0.0
        # Clip to [0,100]
        score = max(0.0, min(100.0, score))
        values.append(score)

    if not labels:
        # empty figure
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Radar needs closing the loop
    labels_loop = labels + [labels[0]]
    values_loop = values + [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_loop,
                theta=labels_loop,
                fill='toself',
                name='SDG relevance'
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=title,
        margin=dict(l=30, r=30, t=60, b=30),
        height=420
    )
    return fig


def summarize_risks(risks: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """
    Convert risks dict to a list of tuples for display: (category, combined_text)
    """
    out = []
    for k in ["environmental", "social", "economic"]:
        items = risks.get(k) or []
        out.append((k.capitalize(), " • ".join(items) if items else "None identified"))
    return out
