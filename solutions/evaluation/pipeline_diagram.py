"""Per-query pipeline flowcharts.

Renders the routing graph as a matplotlib figure with one card per node
(MetaRouting, Intent, Mission, Latency, Decision, FastDecision). Cards
show the model assigned to that step, its output, confidence, latency,
cost, reasoning, and any error / fallback markers. The executed branch
is drawn solid; the skipped branch is dashed grey.

Consumes the flat `evaluation_df` row + `step_df` slice produced by
`solutions.evaluation.frame` (build with `include_reasoning=True`).
"""

from textwrap import shorten, wrap
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from solutions.evaluation.routing_decisions import DEFAULT_TRACE_INDICES


_OK_COLOR = "#2e7d32"        # executed
_SKIP_COLOR = "#9e9e9e"       # skipped branch
_ERROR_COLOR = "#c62828"      # node raised
_FALLBACK_COLOR = "#ef6c00"   # rule-based fallback used


_NODE_W = 0.30
_NODE_H = 0.20
_REASONING_WRAP = 46     # chars per wrapped line inside a node card (tuned for bigger font)
_REASONING_MAX_LINES = 3
_POSITIONS = {
    "meta_routing":  (0.50, 0.85),
    "intent":        (0.50, 0.62),
    "mission":       (0.17, 0.38),
    "latency":       (0.51, 0.38),
    "decision":      (0.34, 0.13),
    "fast_decision": (0.84, 0.38),
}
_SCORING_BOX = (0.01, 0.26, 0.67, 0.24)  # x, y, width, height — wraps mission+latency


def _format_cost(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "—"
    return f"${value:.6f}"


def _format_ms(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.0f} ms"


def _format_score(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.2f}"


def _truncate(text: Optional[str], width: int) -> str:
    if not text:
        return "—"
    return shorten(str(text).replace("\n", " "), width=width, placeholder="…")


def _wrap_reasoning(text: Optional[str]) -> Optional[str]:
    """Wrap reasoning text into a short, fixed-height block that fits in a node card."""
    if not text:
        return None
    cleaned = str(text).replace("\n", " ")
    lines = wrap(cleaned, width=_REASONING_WRAP, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return None
    if len(lines) > _REASONING_MAX_LINES:
        lines = lines[:_REASONING_MAX_LINES]
        lines[-1] = shorten(lines[-1] + " …", width=_REASONING_WRAP, placeholder="…")
    return "\n".join(lines)


def _step_metric(step_rows: pd.DataFrame, step: str, col: str) -> Optional[float]:
    """Return latency_ms or cost for a step name, or None if absent."""
    sub = step_rows[step_rows["step"] == step]
    if sub.empty:
        return None
    return float(sub.iloc[0][col])


def _draw_node(
    ax,
    cx: float,
    cy: float,
    *,
    title: str,
    body_lines: List[Tuple[str, str]],
    border_color: str,
    border_style: str = "solid",
    badge: Optional[str] = None,
    badge_color: Optional[str] = None,
    reasoning: Optional[str] = None,
) -> FancyBboxPatch:
    """Draw a node card centered at (cx, cy). Returns the patch."""
    x = cx - _NODE_W / 2
    y = cy - _NODE_H / 2
    patch = FancyBboxPatch(
        (x, y), _NODE_W, _NODE_H,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.6,
        edgecolor=border_color,
        facecolor="white",
        linestyle=border_style,
    )
    ax.add_patch(patch)

    ax.text(
        cx, y + _NODE_H - 0.022, title,
        ha="center", va="top", fontsize=11, fontweight="bold", color=border_color,
    )

    if badge is not None:
        ax.text(
            x + _NODE_W - 0.008, y + _NODE_H - 0.022, badge,
            ha="right", va="top", fontsize=8, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=badge_color or border_color, edgecolor="none"),
        )

    line_y = y + _NODE_H - 0.055
    for label, value in body_lines:
        ax.text(x + 0.010, line_y, f"{label}:", ha="left", va="top", fontsize=8, color="#555")
        ax.text(x + 0.085, line_y, value, ha="left", va="top", fontsize=8, color="#111")
        line_y -= 0.027

    if reasoning:
        wrapped = _wrap_reasoning(reasoning)
        if wrapped:
            ax.text(
                x + 0.010, y + 0.014, wrapped,
                ha="left", va="bottom", fontsize=7, style="italic", color="#555",
                linespacing=1.15,
            )
    return patch


def _draw_edge(
    ax,
    src: Tuple[float, float],
    dst: Tuple[float, float],
    *,
    executed: bool,
    label: Optional[str] = None,
) -> FancyArrowPatch:
    color = _OK_COLOR if executed else _SKIP_COLOR
    style = "-" if executed else "--"
    sx, sy = src
    dx, dy = dst
    sy = sy - _NODE_H / 2
    dy = dy + _NODE_H / 2
    arrow = FancyArrowPatch(
        (sx, sy), (dx, dy),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linestyle=style,
        linewidth=1.4,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        ax.text(
            mx, my, label,
            ha="center", va="center", fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, linewidth=0.6),
        )
    return arrow


def _draw_scoring_group(ax, *, executed: bool) -> None:
    color = _OK_COLOR if executed else _SKIP_COLOR
    style = "--"
    x, y, w, h = _SCORING_BOX
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        linewidth=1.0,
        edgecolor=color,
        facecolor="none",
        linestyle=style,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.010, y + h - 0.016, "Parallel scoring",
        ha="left", va="top", fontsize=8, color=color, fontstyle="italic",
    )


def render_pipeline_trace(
    eval_row: pd.Series,
    step_rows: pd.DataFrame,
) -> Figure:
    """Render the pipeline flowchart for one (router, query) trace."""
    fast_path = bool(eval_row.get("fast_path", False))

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Title strip
    # ------------------------------------------------------------------
    category = eval_row.get("query_category", "?")
    router = eval_row.get("router", "?")
    query = str(eval_row.get("query", ""))
    title = f"{category} | {router} | {_truncate(query, 80)}"
    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=12, fontweight="bold")

    # ------------------------------------------------------------------
    # Meta routing
    # ------------------------------------------------------------------
    err_meta = bool(eval_row.get("err_meta", False))
    meta_color = _ERROR_COLOR if err_meta else _OK_COLOR
    meta_badge, meta_badge_color = (None, None)
    if err_meta:
        meta_badge, meta_badge_color = "defaults", _FALLBACK_COLOR
    _draw_node(
        ax, *_POSITIONS["meta_routing"],
        title="Meta Routing",
        body_lines=[
            ("intent",   str(eval_row.get("assigned_intent") or "—")),
            ("mission",  str(eval_row.get("assigned_mission") or "—")),
            ("latency",  str(eval_row.get("assigned_latency") or "—")),
            ("decision", str(eval_row.get("assigned_decision") or "—")),
            ("ms / $",   f"{_format_ms(_step_metric(step_rows, 'meta_routing', 'latency_ms'))}  |  "
                          f"{_format_cost(_step_metric(step_rows, 'meta_routing', 'cost'))}"),
        ],
        border_color=meta_color,
        badge=meta_badge,
        badge_color=meta_badge_color,
    )

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------
    err_intent = bool(eval_row.get("err_intent", False))
    intent_color = _ERROR_COLOR if err_intent else _OK_COLOR
    _draw_node(
        ax, *_POSITIONS["intent"],
        title="Intent",
        body_lines=[
            ("model",  str(eval_row.get("assigned_intent") or "—")),
            ("intent", str(eval_row.get("intent") or "—")),
            ("conf", _format_score(eval_row.get("intent_conf"))),
            ("ms / $", f"{_format_ms(_step_metric(step_rows, 'intent', 'latency_ms'))}  |  "
                       f"{_format_cost(_step_metric(step_rows, 'intent', 'cost'))}"),
        ],
        border_color=intent_color,
        badge=("error" if err_intent else None),
        badge_color=_ERROR_COLOR,
        reasoning=eval_row.get("intent_reasoning"),
    )

    # ------------------------------------------------------------------
    # Parallel-scoring group + Mission + Latency
    # ------------------------------------------------------------------
    _draw_scoring_group(ax, executed=not fast_path)

    err_mission = bool(eval_row.get("err_mission", False))
    mission_executed = not fast_path
    mission_color = _ERROR_COLOR if err_mission else (_OK_COLOR if mission_executed else _SKIP_COLOR)
    _draw_node(
        ax, *_POSITIONS["mission"],
        title="Mission",
        body_lines=[
            ("model", str(eval_row.get("assigned_mission") or "—")),
            ("score", _format_score(eval_row.get("mission_score"))),
            ("conf", _format_score(eval_row.get("mission_conf"))),
            ("ms / $", f"{_format_ms(_step_metric(step_rows, 'mission', 'latency_ms'))}  |  "
                       f"{_format_cost(_step_metric(step_rows, 'mission', 'cost'))}"),
        ],
        border_color=mission_color,
        border_style=("solid" if mission_executed or err_mission else "dashed"),
        badge=("error" if err_mission else None),
        badge_color=_ERROR_COLOR,
        reasoning=eval_row.get("mission_reasoning") if mission_executed else None,
    )

    err_latency = bool(eval_row.get("err_latency", False))
    latency_executed = not fast_path
    latency_color = _ERROR_COLOR if err_latency else (_OK_COLOR if latency_executed else _SKIP_COLOR)
    _draw_node(
        ax, *_POSITIONS["latency"],
        title="Latency",
        body_lines=[
            ("model", str(eval_row.get("assigned_latency") or "—")),
            ("score", _format_score(eval_row.get("latency_criticality_score"))),
            ("conf", _format_score(eval_row.get("latency_criticality_conf"))),
            ("ms / $", f"{_format_ms(_step_metric(step_rows, 'latency_criticality', 'latency_ms'))}  |  "
                       f"{_format_cost(_step_metric(step_rows, 'latency_criticality', 'cost'))}"),
        ],
        border_color=latency_color,
        border_style=("solid" if latency_executed or err_latency else "dashed"),
        badge=("error" if err_latency else None),
        badge_color=_ERROR_COLOR,
        reasoning=eval_row.get("latency_criticality_reasoning") if latency_executed else None,
    )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    err_decision = bool(eval_row.get("err_decision", False))
    decision_executed = not fast_path
    if err_decision:
        decision_color, decision_badge, decision_badge_color = _FALLBACK_COLOR, "fallback", _FALLBACK_COLOR
    elif decision_executed:
        decision_color, decision_badge, decision_badge_color = _OK_COLOR, None, None
    else:
        decision_color, decision_badge, decision_badge_color = _SKIP_COLOR, None, None
    _draw_node(
        ax, *_POSITIONS["decision"],
        title="Decision",
        body_lines=[
            ("model",  str(eval_row.get("assigned_decision") or "—")),
            ("pick",   f"{eval_row.get('final_model') or '—'} @ {eval_row.get('final_deploy') or '—'}"),
            ("conf", _format_score(eval_row.get("final_conf"))),
            ("ms / $", f"{_format_ms(_step_metric(step_rows, 'decision', 'latency_ms'))}  |  "
                       f"{_format_cost(_step_metric(step_rows, 'decision', 'cost'))}"),
        ],
        border_color=decision_color,
        border_style=("solid" if decision_executed or err_decision else "dashed"),
        badge=decision_badge,
        badge_color=decision_badge_color,
        reasoning=eval_row.get("final_reasoning") if decision_executed else None,
    )

    # ------------------------------------------------------------------
    # Fast decision
    # ------------------------------------------------------------------
    fast_color = _OK_COLOR if fast_path else _SKIP_COLOR
    _draw_node(
        ax, *_POSITIONS["fast_decision"],
        title="Fast Decision",
        body_lines=[
            ("trigger", "SIMPLE_FACTUAL & high conf"),
            ("pick",   (f"{eval_row.get('final_model') or '—'} @ {eval_row.get('final_deploy') or '—'}"
                        if fast_path else "—")),
            ("conf", _format_score(eval_row.get("final_conf")) if fast_path else "—"),
            ("ms / $", f"{_format_ms(_step_metric(step_rows, 'fast_decision', 'latency_ms'))}  |  —"),
        ],
        border_color=fast_color,
        border_style=("solid" if fast_path else "dashed"),
        reasoning=eval_row.get("final_reasoning") if fast_path else None,
    )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    _draw_edge(ax, _POSITIONS["meta_routing"], _POSITIONS["intent"], executed=True)
    _draw_edge(
        ax, _POSITIONS["intent"], _POSITIONS["mission"],
        executed=not fast_path,
        label=("scoring" if not fast_path else None),
    )
    _draw_edge(
        ax, _POSITIONS["intent"], _POSITIONS["latency"],
        executed=not fast_path,
    )
    _draw_edge(
        ax, _POSITIONS["intent"], _POSITIONS["fast_decision"],
        executed=fast_path,
        label=("fast path" if fast_path else None),
    )
    _draw_edge(ax, _POSITIONS["mission"], _POSITIONS["decision"], executed=not fast_path)
    _draw_edge(ax, _POSITIONS["latency"], _POSITIONS["decision"], executed=not fast_path)

    # ------------------------------------------------------------------
    # Footer legend
    # ------------------------------------------------------------------
    total_ms = eval_row.get("router_ms")
    total_cost = eval_row.get("router_cost")
    footer = (
        f"Routing total: {_format_ms(total_ms)}  |  {_format_cost(total_cost)}"
        f"   ·   path: {'fast' if fast_path else 'full'}"
        f"   ·   errors: {int(eval_row.get('err_count', 0) or 0)}"
    )
    ax.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=9.5, color="#333")

    fig.tight_layout()
    return fig


def render_traces(
    eval_df: pd.DataFrame,
    step_df: pd.DataFrame,
    indices: Optional[List[int]] = None,
    router: Optional[str] = None,
) -> List[Figure]:
    """Render one figure per query index for a given router (custom routers only)."""
    if router is None:
        custom = eval_df[eval_df["assigned_intent"].notna()]
        names = list(dict.fromkeys(custom["router"].tolist()))
        if not names:
            return []
        router = names[0]

    indices = indices if indices is not None else DEFAULT_TRACE_INDICES

    figs: List[Figure] = []
    for idx in indices:
        sub = eval_df[(eval_df["router"] == router) & (eval_df["query_index"] == idx)]
        if sub.empty:
            continue
        eval_row = sub.iloc[0]
        step_rows = step_df[(step_df["router"] == router) & (step_df["query_index"] == idx)]
        figs.append(render_pipeline_trace(eval_row, step_rows))
    return figs
