"""Latency analysis plots — consume the unified evaluation frames.

- Plots 1–5 use the wide `evaluation_df` (uses `router_ms` and `inference_ms`).
- Plots 6–7 use the long `step_df` (filters to wall-time NODE_SEGMENTS to avoid
  double-counting with inner mission / latency_criticality agent durations).
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solutions.evaluation.constants import (
    CATEGORY_ORDER,
    NODE_COLORS,
    NODE_LABELS,
    NODE_SEGMENTS,
    SPLIT_COLORS,
)
from solutions.evaluation.helpers import (
    annotate_small_n,
    filter_and_order_routers,
    mean_std_n,
    router_palette,
)


def plot_inference_overall(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (8, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
        show_errorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean inference latency per router (no router overhead)."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    means, stds, ns = [], [], []
    for name in names:
        vals = df.loc[df["router"] == name, "inference_ms"].to_numpy()
        m, s, n = mean_std_n(vals)
        means.append(m)
        stds.append(s)
        ns.append(n)

    y_pos = np.arange(len(names))
    colors = router_palette(names)[::-1]
    bars = ax.barh(
        y_pos, means[::-1],
        xerr=stds[::-1] if show_errorbar else None,
        capsize=4 if show_errorbar else 0,
        color=colors, edgecolor="black", linewidth=0.5,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean Inference Latency (ms)")
    ax.set_title(title or "Inference-Only Latency by Router")

    rev_means, rev_stds, rev_ns = means[::-1], stds[::-1], ns[::-1]
    err_extent = [s if show_errorbar else 0.0 for s in rev_stds]
    xmax = max((m + e for m, e in zip(rev_means, err_extent)), default=1.0)
    for i, bar in enumerate(bars):
        ax.text(
            rev_means[i] + err_extent[i] + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rev_means[i]:.0f} ms (N={rev_ns[i]})",
            va="center", fontsize=fontsize,
        )

    fig.tight_layout()
    return fig, ax


def plot_inference_per_category(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10, 6),
        fontsize: int = 7,
        title: Optional[str] = None,
        show_errorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean inference latency per (router × category)."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    cats = categories or CATEGORY_ORDER
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(cats))
    total_width = 0.8
    bar_width = total_width / max(len(names), 1)
    colors = router_palette(names)

    for i, name in enumerate(names):
        means, errs, ns = [], [], []
        for cat in cats:
            vals = df.loc[(df["router"] == name) & (df["query_category"] == cat), "inference_ms"].to_numpy()
            m, s, n = mean_std_n(vals)
            means.append(m)
            errs.append(s if n >= 3 else 0.0)
            ns.append(n)

        offsets = x - total_width / 2 + bar_width * (i + 0.5)
        bars = ax.bar(
            offsets, means, bar_width,
            yerr=errs if show_errorbar else None,
            capsize=3 if show_errorbar else 0,
            label=name, color=colors[i], edgecolor="black", linewidth=0.3,
        )
        annotate_small_n(ax, bars, ns, threshold=3, fontsize=fontsize, y_offset_abs=50)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("Query Category")
    ax.set_ylabel("Mean Inference Latency (ms)")
    ax.set_title(title or "Inference-Only Latency by Router and Category")
    ax.legend(fontsize=fontsize + 1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax


def plot_split_overall(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Router overhead + inference per router."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    overhead = [
        float(df.loc[df["router"] == n, "router_ms"].fillna(0).mean()) for n in names
    ]
    inference = [
        float(df.loc[df["router"] == n, "inference_ms"].mean()) for n in names
    ]

    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, overhead, color=SPLIT_COLORS["router_ms"], edgecolor="black", linewidth=0.5, hatch="//",
            label="Router overhead")
    ax.barh(y_pos, inference, left=overhead, color=SPLIT_COLORS["inference_ms"], edgecolor="black", linewidth=0.5,
            label="Inference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_title(title or "Router Overhead vs Inference Latency")
    ax.legend(loc="lower right", fontsize=fontsize)

    xmax = max((o + i for o, i in zip(overhead, inference)), default=1.0)
    for yp, o, inf in zip(y_pos, overhead, inference):
        if o > 0:
            ax.text(o + xmax * 0.005, yp, f"{o:.0f}", va="center", fontsize=fontsize - 1, color="black")
        ax.text(o + inf + xmax * 0.01, yp, f"total {o + inf:.0f} ms", va="center", fontsize=fontsize - 1)

    fig.tight_layout()
    return fig, ax


def plot_split_overall_pct(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overhead as percent of total latency."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    overhead_pct, inference_pct, totals = [], [], []
    for name in names:
        sub = df[df["router"] == name]
        o = float(sub["router_ms"].fillna(0).mean())
        i = float(sub["inference_ms"].mean())
        total = o + i if (o + i) > 0 else 1.0
        overhead_pct.append(o / total * 100)
        inference_pct.append(i / total * 100)
        totals.append(o + i)

    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, overhead_pct, color=SPLIT_COLORS["router_ms"], edgecolor="black", linewidth=0.5, hatch="//",
            label="Router overhead")
    ax.barh(y_pos, inference_pct, left=overhead_pct, color=SPLIT_COLORS["inference_ms"], edgecolor="black",
            linewidth=0.5, label="Inference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Percent of Total (%)")
    ax.set_xlim(0, 105)
    ax.set_title(title or "Router Overhead vs Inference (percent of total)")
    ax.legend(loc="lower right", fontsize=fontsize)

    for yp, o_pct, total in zip(y_pos, overhead_pct, totals):
        ax.text(o_pct + 1, yp, f"{o_pct:.1f}% overhead · total {total:.0f} ms", va="center", fontsize=fontsize - 1)

    fig.tight_layout()
    return fig, ax


def plot_split_per_category(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (15, 5),
        fontsize: int = 7,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """One subplot per category, stacked bars (overhead + inference) per router."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    cats = categories or CATEGORY_ORDER

    fig, axes = plt.subplots(1, len(cats), figsize=figsize, sharey=True)
    if len(cats) == 1:
        axes = np.array([axes])

    for ax, cat in zip(axes, cats):
        x = np.arange(len(names))
        overhead = [
            float(df.loc[(df["router"] == n) & (df["query_category"] == cat), "router_ms"].fillna(0).mean())
            for n in names
        ]
        inference = [
            float(df.loc[(df["router"] == n) & (df["query_category"] == cat), "inference_ms"].mean())
            for n in names
        ]
        ax.bar(x, overhead, color=SPLIT_COLORS["router_ms"], edgecolor="black", linewidth=0.4, hatch="//",
               label="Router overhead")
        ax.bar(x, inference, bottom=overhead, color=SPLIT_COLORS["inference_ms"], edgecolor="black", linewidth=0.4,
               label="Inference")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=fontsize)
        ax.set_title(cat, fontsize=fontsize + 2)

    axes[0].set_ylabel("Mean Latency (ms)")
    axes[-1].legend(fontsize=fontsize + 1, loc="upper right")
    fig.suptitle(title or "Router Overhead vs Inference by Category", y=1.02)
    fig.tight_layout()
    return fig, axes


def _node_pivot(
        step_df: pd.DataFrame,
        names: List[str],
        category: Optional[str] = None,
        value: str = "latency_ms",
) -> pd.DataFrame:
    """Mean value per (router, step)."""
    mask = step_df["step"].isin(NODE_SEGMENTS) & step_df["router"].isin(names)
    if category is not None:
        mask = mask & (step_df["query_category"] == category)
    pivot = step_df.loc[mask].groupby(["router", "step"])[value].mean().unstack(fill_value=0)
    return pivot.reindex(index=names, columns=NODE_SEGMENTS, fill_value=0)


def plot_node_breakdown_overall(
        step_df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-node routing latency per custom router."""
    names = filter_and_order_routers(step_df, exclude_routers, router_order)
    pivot = _node_pivot(step_df, names)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for seg in NODE_SEGMENTS:
        values = pivot[seg].values
        ax.barh(y_pos, values, left=left, color=NODE_COLORS[seg], edgecolor="black", linewidth=0.3,
                label=NODE_LABELS[seg])
        for yp, v, l in zip(y_pos, values, left):
            if v > 0:
                ax.text(l + v / 2, yp, f"{v:.0f}", va="center", ha="center", fontsize=fontsize - 2)
        left = left + values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Routing Latency (ms)")
    ax.set_title(title or "Routing Latency Breakdown by Node (custom routers)")
    ax.legend(fontsize=fontsize, loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_node_breakdown_per_category(
        step_df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (15, 5),
        fontsize: int = 7,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """One subplot per category, stacked node-breakdown bars per custom router."""
    names = filter_and_order_routers(step_df, exclude_routers, router_order)
    cats = categories or CATEGORY_ORDER

    fig, axes = plt.subplots(1, len(cats), figsize=figsize, sharey=True)
    if len(cats) == 1:
        axes = np.array([axes])

    for ax, cat in zip(axes, cats):
        pivot = _node_pivot(step_df, names, category=cat)
        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for seg in NODE_SEGMENTS:
            values = pivot[seg].values
            ax.bar(x, values, bottom=bottom, color=NODE_COLORS[seg], edgecolor="black", linewidth=0.3,
                   label=NODE_LABELS[seg])
            bottom = bottom + values

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=fontsize)
        ax.set_title(cat, fontsize=fontsize + 2)

    axes[0].set_ylabel("Mean Routing Latency (ms)")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), fontsize=fontsize + 1, loc="upper right")
    fig.suptitle(
        title or "Routing Latency Breakdown by Node and Category (custom routers)",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes
