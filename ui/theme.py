"""Visual tokens. Chart palette follows the validated reference palette:
categorical slots in fixed order, status colours reserved for risk/health and
always paired with a text label + icon."""

from __future__ import annotations

import plotly.graph_objects as go

INK = "#0F172A"
INK_2 = "#475569"
MUTED = "#898781"
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
SURFACE = "#FFFFFF"
NAVY = "#0B2545"
TEAL = "#0F766E"

SERIES = ["#2A78D6", "#EB6834", "#1BAF7A"]  # blue, orange, aqua (validated all-pairs)
SEQUENTIAL = ["#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B"]

RISK = {
    "high": {"color": "#D03B3B", "bg": "#FDECEC", "icon": "▲", "label": "High"},
    "medium": {"color": "#B7791F", "bg": "#FEF6E4", "icon": "◆", "label": "Medium"},
    "low": {"color": "#0B7A0B", "bg": "#E8F6E8", "icon": "●", "label": "Low"},
}
RISK_BAR = {"high": "#D03B3B", "medium": "#FAB219", "low": "#0CA30C"}
STATUS = {
    "stable": ("#0B7A0B", "●", "Stable"), "ok": ("#0B7A0B", "●", "OK"),
    "warning": ("#B7791F", "◆", "Warning"), "alert": ("#D03B3B", "▲", "Alert"),
    "medium": ("#B7791F", "◆", "Medium"), "high": ("#D03B3B", "▲", "High"),
}


def layout(fig: go.Figure, *, height: int = 320, title: str | None = None,
           legend: bool = False, xtitle: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        template="none", height=height, margin=dict(l=8, r=8, t=40 if title else 12, b=8),
        title=dict(text=title, x=0, font=dict(size=14, color=INK)) if title else None,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=legend,
        legend=dict(orientation="h", x=0, y=-0.22 if xtitle else -0.12, yanchor="top",
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=INK, size=12)),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside",
                     tickfont=dict(color=MUTED), title=dict(text=xtitle, font=dict(color=INK_2)),
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title=dict(text=ytitle, font=dict(color=INK_2)), automargin=True)
    return fig
