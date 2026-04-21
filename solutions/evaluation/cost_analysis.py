"""Cost analysis plots — mirror the latency analysis plot set for USD cost."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solutions.evaluation.constants import (
    CATEGORY_ORDER,
    COST_SEGMENTS,
    NODE_COLORS,
    NODE_LABELS,
    SPLIT_COLORS,
)
from solutions.evaluation.helpers import (
    annotate_small_n,
    filter_and_order_routers,
    mean_std_n,
    router_palette,
)


def _format_cost(v: float) -> str:
    return f"${v:.6f}"


def plot_cost_inference_overall(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (8, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
        show_errorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean inference cost per router (no router overhead)."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    means, stds, ns = [], [], []
    for name in names:
        vals = df.loc[df["router"] == name, "inference_cost"].to_numpy()
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
    ax.set_xlabel("Mean Inference Cost (USD)")
    ax.set_title(title or "Inference-Only Cost by Router")

    rev_means, rev_stds, rev_ns = means[::-1], stds[::-1], ns[::-1]
    err_extent = [s if show_errorbar else 0.0 for s in rev_stds]
    xmax = max((m + e for m, e in zip(rev_means, err_extent)), default=1.0)
    for i, bar in enumerate(bars):
        ax.text(
            rev_means[i] + err_extent[i] + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{_format_cost(rev_means[i])} (N={rev_ns[i]})",
            va="center", fontsize=fontsize,
        )

    fig.tight_layout()
    return fig, ax


def plot_cost_inference_per_category(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10, 6),
        fontsize: int = 7,
        title: Optional[str] = None,
        show_errorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean inference cost per (router × category)."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    cats = categories or CATEGORY_ORDER
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(cats))
    total_width = 0.8
    bar_width = total_width / max(len(names), 1)
    colors = router_palette(names)

    # Scan all means to size y_offset for n-labels
    all_means = []

    for i, name in enumerate(names):
        means, errs, ns = [], [], []
        for cat in cats:
            vals = df.loc[(df["router"] == name) & (df["query_category"] == cat), "inference_cost"].to_numpy()
            m, s, n = mean_std_n(vals)
            means.append(m)
            errs.append(s if n >= 3 else 0.0)
            ns.append(n)
            all_means.append(m)

        offsets = x - total_width / 2 + bar_width * (i + 0.5)
        bars = ax.bar(
            offsets, means, bar_width,
            yerr=errs if show_errorbar else None,
            capsize=3 if show_errorbar else 0,
            label=name, color=colors[i], edgecolor="black", linewidth=0.3,
        )
        # y_offset_abs in USD]
        y_abs = max(all_means) * 0.02 if all_means else 0.00001
        annotate_small_n(ax, bars, ns, threshold=3, fontsize=fontsize, y_offset_abs=y_abs)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("Query Category")
    ax.set_ylabel("Mean Inference Cost (USD)")
    ax.set_title(title or "Inference-Only Cost by Router and Category")
    ax.legend(fontsize=fontsize + 1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax


def plot_cost_total_overall(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (8, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Mean total cost (inference + router overhead) per router."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    totals = []
    for name in names:
        sub = df[df["router"] == name]
        totals.append(float(sub["router_cost"].fillna(0).mean() + sub["inference_cost"].mean()))

    y_pos = np.arange(len(names))
    colors = router_palette(names)[::-1]
    bars = ax.barh(y_pos, totals[::-1], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean Total Cost (USD)")
    ax.set_title(title or "Total Cost by Router (inference + routing)")

    rev_totals = totals[::-1]
    xmax = max(rev_totals, default=1.0)
    for i, bar in enumerate(bars):
        ax.text(
            rev_totals[i] + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            _format_cost(rev_totals[i]),
            va="center", fontsize=fontsize,
        )

    fig.tight_layout()
    return fig, ax


def plot_cost_split_overall_pct(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Router overhead vs inference as percent of total."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    overhead_pct, inference_pct, totals = [], [], []
    for name in names:
        sub = df[df["router"] == name]
        o = float(sub["router_cost"].fillna(0).mean())
        i = float(sub["inference_cost"].mean())
        total = o + i if (o + i) > 0 else 1.0
        overhead_pct.append(o / total * 100)
        inference_pct.append(i / total * 100)
        totals.append(o + i)

    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, overhead_pct, color=SPLIT_COLORS["router_cost"],
            edgecolor="black", linewidth=0.5, hatch="//", label="Router overhead")
    ax.barh(y_pos, inference_pct, left=overhead_pct, color=SPLIT_COLORS["inference_cost"],
            edgecolor="black", linewidth=0.5, label="Inference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Percent of Total (%)")
    ax.set_xlim(0, 120)
    ax.set_title(title or "Router Overhead vs Inference Cost (percent of total)")
    ax.legend(loc="lower right", fontsize=fontsize)

    for yp, o_pct, total in zip(y_pos, overhead_pct, totals):
        ax.text(101, yp, f"{o_pct:.1f}% overhead · total {_format_cost(total)}", va="center", fontsize=fontsize - 1)

    fig.tight_layout()
    return fig, ax


def plot_cost_split_overall(
        df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Router overhead (hatched) + inference cost."""
    names = filter_and_order_routers(df, exclude_routers, router_order)
    fig, ax = plt.subplots(figsize=figsize)

    overhead = [
        float(df.loc[df["router"] == n, "router_cost"].fillna(0).mean()) for n in names
    ]
    inference = [
        float(df.loc[df["router"] == n, "inference_cost"].mean()) for n in names
    ]

    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, overhead, color=SPLIT_COLORS["router_cost"], edgecolor="black",
            linewidth=0.5, hatch="//", label="Router overhead")
    ax.barh(y_pos, inference, left=overhead, color=SPLIT_COLORS["inference_cost"],
            edgecolor="black", linewidth=0.5, label="Inference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Cost (USD)")
    ax.set_title(title or "Router Overhead vs Inference Cost")
    ax.legend(loc="lower right", fontsize=fontsize)

    xmax = max((o + i for o, i in zip(overhead, inference)), default=1.0)
    for yp, o, inf in zip(y_pos, overhead, inference):
        ax.text(o + inf + xmax * 0.01, yp, f"total {_format_cost(o + inf)}", va="center", fontsize=fontsize - 1)

    fig.tight_layout()
    return fig, ax


def plot_cost_split_per_category(
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
            float(df.loc[(df["router"] == n) & (df["query_category"] == cat), "router_cost"].fillna(0).mean())
            for n in names
        ]
        inference = [
            float(df.loc[(df["router"] == n) & (df["query_category"] == cat), "inference_cost"].mean())
            for n in names
        ]
        ax.bar(x, overhead, color=SPLIT_COLORS["router_cost"], edgecolor="black",
               linewidth=0.4, hatch="//", label="Router overhead")
        ax.bar(x, inference, bottom=overhead, color=SPLIT_COLORS["inference_cost"],
               edgecolor="black", linewidth=0.4, label="Inference")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=fontsize)
        ax.set_title(cat, fontsize=fontsize + 2)

    axes[0].set_ylabel("Mean Cost (USD)")
    axes[-1].legend(fontsize=fontsize + 1, loc="upper right")
    fig.suptitle(title or "Router Overhead vs Inference Cost by Category", y=1.02)
    fig.tight_layout()
    return fig, axes


def _cost_node_pivot(
        step_df: pd.DataFrame,
        names: List[str],
        category: Optional[str] = None,
) -> pd.DataFrame:
    """Mean cost per (router, step) pivoted to wide"""
    mask = step_df["step"].isin(COST_SEGMENTS) & step_df["router"].isin(names)
    if category is not None:
        mask = mask & (step_df["query_category"] == category)
    pivot = step_df.loc[mask].groupby(["router", "step"])["cost"].mean().unstack(fill_value=0)
    return pivot.reindex(index=names, columns=COST_SEGMENTS, fill_value=0)


def plot_cost_node_breakdown_overall(
        step_df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (9, 6),
        fontsize: int = 8,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-agent routing cost per custom router.
    Custom routers only — baselines have no step_df rows.
    """
    names = filter_and_order_routers(step_df, exclude_routers, router_order)
    pivot = _cost_node_pivot(step_df, names)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for seg in COST_SEGMENTS:
        values = pivot[seg].values
        ax.barh(y_pos, values, left=left, color=NODE_COLORS[seg], edgecolor="black", linewidth=0.3,
                label=NODE_LABELS[seg])
        left = left + values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Routing Cost (USD)")
    ax.set_title(title or "Routing Cost Breakdown by Agent (custom routers)")
    ax.legend(fontsize=fontsize, loc="lower right")

    # Annotate totals at the bar ends
    totals = pivot.sum(axis=1).values
    xmax = max(totals, default=1.0)
    for yp, t in zip(y_pos, totals):
        ax.text(t + xmax * 0.02, yp, _format_cost(t), va="center", fontsize=fontsize - 1)

    fig.tight_layout()
    return fig, ax


def plot_cost_node_breakdown_per_category(
        step_df: pd.DataFrame,
        exclude_routers: Optional[List[str]] = None,
        router_order: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (15, 5),
        fontsize: int = 7,
        title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """One subplot per category, stacked per-agent cost bars per custom router."""
    names = filter_and_order_routers(step_df, exclude_routers, router_order)
    cats = categories or CATEGORY_ORDER

    fig, axes = plt.subplots(1, len(cats), figsize=figsize, sharey=True)
    if len(cats) == 1:
        axes = np.array([axes])

    for ax, cat in zip(axes, cats):
        pivot = _cost_node_pivot(step_df, names, category=cat)
        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for seg in COST_SEGMENTS:
            values = pivot[seg].values
            ax.bar(x, values, bottom=bottom, color=NODE_COLORS[seg], edgecolor="black", linewidth=0.3,
                   label=NODE_LABELS[seg])
            bottom = bottom + values

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=fontsize)
        ax.set_title(cat, fontsize=fontsize + 2)

    axes[0].set_ylabel("Mean Routing Cost (USD)")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), fontsize=fontsize + 1, loc="upper right")
    fig.suptitle(
        title or "Routing Cost Breakdown by Agent and Category (custom routers)",
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes
