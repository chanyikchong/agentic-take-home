"""Routing decision tables — consume the unified evaluation frame."""

from collections import Counter
from pathlib import Path
from textwrap import wrap as _text_wrap
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from solutions.evaluation.constants import CATEGORY_ORDER, MONITOR_TO_ROUTER
from solutions.evaluation.helpers import filter_and_order_routers


_DISPLAY_COLS = [
    "category", "router", "model_selections", "edge_deployment", "cloud_deployment",
    "tier_mode", "tier_agreement",
    "mean_quality", "mean_latency_ms", "mean_cost",
    "mean_mission_score", "mean_latency_score",
]
_FMT_MAP = {
    "tier_agreement": "{:.0%}",
    "mean_quality": "{:.2f}",
    "mean_latency_ms": "{:.0f}",
    "mean_cost": "${:.6f}",
    "mean_mission_score": "{:.2f}",
    "mean_latency_score": "{:.2f}",
}
_HIGHER_BETTER = ("mean_quality", "tier_agreement")
_LOWER_BETTER = ("mean_latency_ms", "mean_cost")

# Column chars-cap for string wrapping. Columns not listed are never wrapped.
_WRAP_MAX_CHARS = {"model_selections": 36}


DEFAULT_TRACE_INDICES = [0, 3, 6, 9, 12]

OVERALL_LABEL = "overall"


def _model_selection_summary(model_keys: List[str]) -> str:
    """Format model picks as count-aggregated string.

    E.g. ["gemma-3n-e4b", "gemma-3n-e4b", "gemma-3-27b"] -> "gemma-3n-e4b (2), gemma-3-27b (1)"
    """
    counts = Counter(model_keys)
    parts = [f"{model} ({count})" for model, count in counts.most_common()]
    return ", ".join(parts)


def _tier_mode_and_agreement(tiers: List[str]) -> Tuple[str, float]:
    if not tiers:
        return ("", 0.0)
    counts = Counter(tiers)
    mode, mode_count = counts.most_common(1)[0]
    return mode, mode_count / len(tiers)


def _has_monitor(sub: pd.DataFrame) -> bool:
    """True iff the router backing `sub` has monitor data (custom router)."""
    return sub["assigned_intent"].notna().any()


def _aggregate_cell(sub: pd.DataFrame) -> dict:
    """Aggregate a (router, category) or (router, overall) slice into metric columns."""
    completed = sub[~sub["timed_out"]]
    n_completed = len(completed)

    model_keys = completed["model_key"].tolist()
    tiers = completed["model_tier"].tolist()
    scores = completed["quality_score"].dropna().tolist()

    model_summary = _model_selection_summary(model_keys) if model_keys else ""
    edge_deployment = int((completed["deployment"] == "edge").sum())
    cloud_deployment = int((completed["deployment"] == "cloud").sum())
    tier_mode, tier_agreement = _tier_mode_and_agreement(tiers)

    mean_quality = float(np.mean(scores)) if scores else np.nan
    mean_latency = float(completed["inference_ms"].mean()) if n_completed else np.nan
    mean_cost = float(completed["inference_cost"].mean()) if n_completed else np.nan

    if _has_monitor(completed):
        mean_mission = float(completed["mission_score"].mean())
        mean_latency_score = float(completed["latency_criticality_score"].mean())
        fast_path_count = int(completed["fast_path"].sum())
    else:
        mean_mission = np.nan
        mean_latency_score = np.nan
        fast_path_count = np.nan

    return {
        "n_completed": n_completed,
        "model_selections": model_summary,
        "edge_deployment": edge_deployment,
        "cloud_deployment": cloud_deployment,
        "tier_mode": tier_mode,
        "tier_agreement": tier_agreement,
        "mean_quality": mean_quality,
        "mean_latency_ms": mean_latency,
        "mean_cost": mean_cost,
        "mean_mission_score": mean_mission,
        "mean_latency_score": mean_latency_score,
        "fast_path_count": fast_path_count,
    }


def build_routing_decisions_table(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    router_order: Optional[List[str]] = None,
    exclude_routers: Optional[List[str]] = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Summary table of routing decisions per (category, router).

    When `include_overall=True` (default), an additional row per router is
    appended with `category="overall"` aggregating across all categories.

    Baselines get NaN for monitor-derived columns
    (mean_mission_score, mean_latency_score, fast_path_count).
    """
    cats = categories or CATEGORY_ORDER
    names = filter_and_order_routers(df, exclude_routers, router_order)

    rows = []
    for cat in cats:
        for router_name in names:
            sub = df[(df["router"] == router_name) & (df["query_category"] == cat)]
            rows.append({"category": cat, "router": router_name, **_aggregate_cell(sub)})

    if include_overall:
        for router_name in names:
            sub = df[df["router"] == router_name]
            rows.append({"category": OVERALL_LABEL, "router": router_name, **_aggregate_cell(sub)})

    return pd.DataFrame(rows)


def _prepare_display_frame(
    df: pd.DataFrame,
    *,
    group_by: str = "category",
    categories: Optional[List[str]] = None,
    routers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter, select display columns, and sort as both renderers expect."""
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    if categories:
        out = out[out["category"].isin(categories)]
    if routers:
        out = out[out["router"].isin(routers)]
    cat_order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    cat_order[OVERALL_LABEL] = len(CATEGORY_ORDER)
    out["_cat_sort"] = out["category"].map(cat_order)
    if group_by == "category":
        out = out.sort_values(["_cat_sort", "router"])
    else:
        out = out.sort_values(["router", "_cat_sort"])
    return out.drop(columns=["_cat_sort"]).reset_index(drop=True)


def style_routing_decisions_table(
    df: pd.DataFrame,
    group_by: str = "category",
    categories: Optional[List[str]] = None,
    routers: Optional[List[str]] = None,
) -> Styler:
    """Format the routing decisions DataFrame with gradients for quick scanning.

    Args:
        df:         DataFrame from build_routing_decisions_table().
        group_by:   "category" (default) or "router" — controls row ordering.
        categories: Filter to these categories only.
        routers:    Filter to these routers only.
    """
    out = _prepare_display_frame(df, group_by=group_by, categories=categories, routers=routers)
    fmt = {k: v for k, v in _FMT_MAP.items() if k in out.columns}
    styler = out.style.format(fmt, na_rep="—")

    higher_better = [c for c in _HIGHER_BETTER if c in out.columns]
    if higher_better:
        styler = styler.background_gradient(subset=higher_better, cmap="RdYlGn", vmin=0)
    lower_better = [c for c in _LOWER_BETTER if c in out.columns]
    if lower_better:
        styler = styler.background_gradient(subset=lower_better, cmap="RdYlGn_r")

    # Bold the overall rows so they stand out from per-category rows.
    def _bold_overall(row: pd.Series) -> List[str]:
        if row.get("category") == OVERALL_LABEL:
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)
    styler = styler.apply(_bold_overall, axis=1)

    styler = styler.hide(axis="index")
    return styler


def _gradient_color(value: float, vmin: float, vmax: float, reverse: bool) -> Tuple[float, float, float, float]:
    """Return an RGBA color for a numeric cell using the RdYlGn gradient."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    cmap = mpl.colormaps["RdYlGn_r" if reverse else "RdYlGn"]
    # Soften colors so black text stays readable.
    r, g, b, _ = cmap(t)
    return (r, g, b, 0.55)


def save_routing_decisions_image(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    group_by: str = "category",
    categories: Optional[List[str]] = None,
    routers: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    title: Optional[str] = None,
) -> Path:
    """Render the routing-decisions table to a PNG (matplotlib — no new deps).

    Mirrors `style_routing_decisions_table`: same sort order, value formatting,
    RdYlGn gradient on `mean_quality` / `tier_agreement` and reversed on
    `mean_latency_ms` / `mean_cost`, bold rows for `category="overall"`.

    Args:
        df:         DataFrame from `build_routing_decisions_table()`.
        path:       Output file path (`.png` recommended).
        group_by:   "category" or "router" (row ordering).
        categories: Filter to these categories only.
        routers:    Filter to these routers only.
        figsize:    (width, height) inches. If None, auto-sized by row/col count.
        dpi:        Save DPI.
        title:      Optional figure title.

    Returns:
        The output Path.
    """
    out = _prepare_display_frame(df, group_by=group_by, categories=categories, routers=routers)
    return _render_table_image(
        out, path,
        fmt_map=_FMT_MAP,
        higher_better=_HIGHER_BETTER,
        lower_better=_LOWER_BETTER,
        wrap_max_chars=_WRAP_MAX_CHARS,
        bold_row_predicate=lambda row: row.get("category") == OVERALL_LABEL,
        title=title,
        figsize=figsize,
        dpi=dpi,
    )


def _wrap_cell_text(text: str, cap: int) -> str:
    """Wrap a cell string on comma boundaries, falling back to character wrap."""
    if not cap or not isinstance(text, str) or len(text) <= cap:
        return text
    parts = [p.strip() for p in text.split(",")]
    lines, current = [], ""
    for p in parts:
        candidate = (current + (", " if current else "") + p)
        if len(candidate) <= cap:
            current = candidate
        else:
            if current:
                lines.append(current + ",")
            current = p
    if current:
        lines.append(current)
    out_lines: List[str] = []
    for ln in lines:
        if len(ln) <= cap:
            out_lines.append(ln)
        else:
            out_lines.extend(_text_wrap(ln, width=cap, break_long_words=False, break_on_hyphens=False))
    return "\n".join(out_lines)


def _render_table_image(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    fmt_map: Dict[str, str],
    higher_better: Tuple[str, ...] = (),
    lower_better: Tuple[str, ...] = (),
    wrap_max_chars: Optional[Dict[str, int]] = None,
    bold_row_predicate: Optional[callable] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    fontsize: int = 8,
) -> Path:
    """Render a DataFrame to a PNG using matplotlib.

    Shared body for `save_routing_decisions_image` and `save_pipeline_trace_image`.
    Formats cells by `fmt_map`, gradients `higher_better` / `lower_better` columns,
    wraps long strings per `wrap_max_chars`, bolds rows matching `bold_row_predicate`.
    Column widths and row heights auto-adapt to content.
    """
    wrap_max_chars = wrap_max_chars or {}
    cols = df.columns.tolist()

    # Gradient ranges on numeric source values, not formatted strings.
    higher_cols = [c for c in higher_better if c in cols]
    lower_cols = [c for c in lower_better if c in cols]
    col_ranges: Dict[str, Tuple[float, float]] = {}
    for c in higher_cols + lower_cols:
        values = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        if values.size:
            vmin = 0.0 if c in higher_cols else float(values.min())
            col_ranges[c] = (vmin, float(values.max()))

    def _format(col: str, value) -> str:
        return _format_cell_with_fmt(value, fmt_map.get(col))

    cell_text = [
        [_wrap_cell_text(_format(c, df.iloc[i][c]), wrap_max_chars.get(c, 0)) for c in cols]
        for i in range(len(df))
    ]

    white = (1.0, 1.0, 1.0, 1.0)
    cell_colors = [[white] * len(cols) for _ in range(len(df))]
    for c in higher_cols + lower_cols:
        if c not in col_ranges:
            continue
        vmin, vmax = col_ranges[c]
        j = cols.index(c)
        for i in range(len(df)):
            v = pd.to_numeric(df.iloc[i][c], errors="coerce")
            if pd.isna(v):
                continue
            cell_colors[i][j] = _gradient_color(float(v), vmin, vmax, reverse=(c in lower_cols))

    n_rows = len(df)
    n_cols = len(cols)

    # Content-aware column widths.
    char_in = fontsize * 0.58 / 72.0
    pad_in = 0.20
    min_col_in = 0.55
    max_line_len: List[int] = []
    for j, c in enumerate(cols):
        longest = len(str(c))
        for row in cell_text:
            for line in str(row[j]).split("\n"):
                if len(line) > longest:
                    longest = len(line)
        max_line_len.append(longest)
    col_widths_in = [max(min_col_in, n * char_in + pad_in) for n in max_line_len]

    row_line_counts = [max(1, *(len(str(cell).split("\n")) for cell in row)) for row in cell_text]
    line_h_in = 0.20
    height_in = line_h_in * (1 + sum(row_line_counts)) + (0.35 if title else 0) + 0.30
    width_in = sum(col_widths_in) + 0.25
    if figsize is None:
        figsize = (width_in, height_in)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")

    header_color = "#2e7d32"
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellColours=cell_colors,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    total_w = sum(col_widths_in)
    for j in range(n_cols):
        frac = col_widths_in[j] / total_w
        for i in range(n_rows + 1):
            table[(i, j)].set_width(frac)

    total_lines = 1 + sum(row_line_counts)
    for j in range(n_cols):
        table[(0, j)].set_height(1.0 / total_lines)
    for i, lc in enumerate(row_line_counts):
        for j in range(n_cols):
            table[(i + 1, j)].set_height(lc / total_lines)

    for j in range(n_cols):
        hdr = table[(0, j)]
        hdr.set_facecolor(header_color)
        hdr.set_text_props(color="white", fontweight="bold")

    if bold_row_predicate is not None:
        for i in range(n_rows):
            row_series = df.iloc[i]
            if bold_row_predicate(row_series):
                for j in range(n_cols):
                    cell = table[(i + 1, j)]
                    cell.set_text_props(fontweight="bold")
                    cell.set_edgecolor("#444")
                    cell.set_linewidth(0.8)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _format_cell_with_fmt(value, fmt: Optional[str]) -> str:
    """Format using an explicit format string (falls back to str / '—' for NaN)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if fmt is None:
        return str(value)
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Pipeline-trace image
# ---------------------------------------------------------------------------

_PIPELINE_FMT_MAP = {
    "intent_confidence": "{:.2f}",
    "mission_score": "{:.2f}",
    "mission_confidence": "{:.2f}",
    "latency_score": "{:.2f}",
    "latency_confidence": "{:.2f}",
    "decision_confidence": "{:.2f}",
    "quality_score": "{:.2f}",
    "routing_latency_ms": "{:.0f}",
    "routing_cost": "${:.6f}",
    "fast_path": "{}",
    "err_count": "{:d}",
}

_PIPELINE_HIGHER_BETTER = ("quality_score", "intent_confidence", "mission_confidence",
                           "latency_confidence", "decision_confidence")
_PIPELINE_LOWER_BETTER = ("routing_latency_ms", "routing_cost", "err_count")
_PIPELINE_WRAP_CAPS = {"query": 42, "decision_reasoning": 48}


def save_pipeline_trace_image(
    pipeline_df: pd.DataFrame,
    path: Union[str, Path],
    *,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    title: Optional[str] = None,
) -> Path:
    """Render the per-query pipeline trace table to PNG.

    Input is the output of `build_pipeline_trace_table`. Gradients:
      - higher-better: quality_score, intent_confidence, mission_confidence,
        latency_confidence, decision_confidence
      - lower-better: routing_latency_ms, routing_cost, err_count
    Long query and decision_reasoning cells are wrapped.
    Rows where `fast_path=True` are rendered bold to highlight the shortcut.
    """
    return _render_table_image(
        pipeline_df, path,
        fmt_map=_PIPELINE_FMT_MAP,
        higher_better=_PIPELINE_HIGHER_BETTER,
        lower_better=_PIPELINE_LOWER_BETTER,
        wrap_max_chars=_PIPELINE_WRAP_CAPS,
        bold_row_predicate=lambda row: bool(row.get("fast_path", False)),
        title=title,
        figsize=figsize,
        dpi=dpi,
    )


def build_pipeline_trace_table(
    df: pd.DataFrame,
    query_indices: Optional[List[int]] = None,
    router_order: Optional[List[str]] = None,
    exclude_routers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Detailed per-query trace across custom routers.

    Pulls the full pipeline state for selected `query_indices` (positional),
    one row per (query_index, router). Baselines excluded (no pipeline).
    Reasoning columns included when they exist on the frame
    (i.e. when the frame was built with `include_reasoning=True`).
    """
    indices = query_indices or DEFAULT_TRACE_INDICES
    # Custom routers only — identified by non-null assigned_intent
    custom_df = df[df["assigned_intent"].notna()]
    names = filter_and_order_routers(custom_df, exclude_routers, router_order)

    has_reasoning = "final_reasoning" in df.columns

    rows = []
    for idx in indices:
        for router_name in names:
            sub = custom_df[
                (custom_df["router"] == router_name) & (custom_df["query_index"] == idx)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            row = {
                "category": r["query_category"],
                "query": r["query"],
                "router": router_name,
                "intent_model": r["assigned_intent"],
                "mission_model": r["assigned_mission"],
                "latency_model": r["assigned_latency"],
                "decision_model_agent": r["assigned_decision"],
                "intent": r["intent"],
                "intent_confidence": r["intent_conf"],
                "mission_score": r["mission_score"],
                "mission_confidence": r["mission_conf"],
                "latency_score": r["latency_criticality_score"],
                "latency_confidence": r["latency_criticality_conf"],
                "final_model": r["final_model"],
                "final_deployment": r["final_deploy"],
                "decision_confidence": r["final_conf"],
                "quality_score": r["quality_score"],
                "fast_path": r["fast_path"],
                "routing_latency_ms": r["router_ms"],
                "routing_cost": r["router_cost"],
                "err_count": r["err_count"],
            }
            if has_reasoning:
                row["decision_reasoning"] = r.get("final_reasoning")
            rows.append(row)

    return pd.DataFrame(rows)
