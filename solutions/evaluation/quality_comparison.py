"""Quality comparison plots — consume the unified evaluation frame."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solutions.evaluation.constants import CATEGORY_ORDER, ROUTER_CMAP
from solutions.evaluation.frame import expand_to_all_queries
from solutions.evaluation.helpers import (
    annotate_small_n,
    apply_quality_penalty,
    filter_and_order_routers,
    mean_std_n,
    router_palette,
)


_QUALITY_COL = "quality_score"
_QUALITY_COL_PENALIZED = "quality_effective"


def _prepare(
    df: pd.DataFrame,
    penalize_failures: bool,
    total_queries: Optional[int] = None,
    query_catalog: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str]:
    """Return (df_to_aggregate, quality_column_name)."""
    if penalize_failures:
        padded = expand_to_all_queries(df, total_queries=total_queries, query_catalog=query_catalog)
        return apply_quality_penalty(padded), _QUALITY_COL_PENALIZED
    return df, _QUALITY_COL


def plot_overall_quality(
    df: pd.DataFrame,
    exclude_routers: Optional[List[str]] = None,
    router_order: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (8, 6),
    title: Optional[str] = None,
    fontsize: Optional[int] = 6,
    show_errorbar: bool = True,
    penalize_failures: bool = False,
    total_queries: Optional[int] = None,
    query_catalog: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of mean quality score per router.

    When `penalize_failures=True`, missing queries are padded with quality 0
    and timed-out rows are forced to 0 (judge-NaN rows on completed responses
    still drop out). The zero-padded contribution to each bar is drawn as a
    hatched overlay so the reader can see how much of the mean is "real".
    """
    names = filter_and_order_routers(df, exclude_routers, router_order)
    prepared, qcol = _prepare(df, penalize_failures, total_queries, query_catalog)
    fig, ax = plt.subplots(figsize=figsize)

    means, stds, annotations, raw_means = [], [], [], []
    for name in names:
        sub = prepared[prepared["router"] == name]
        vals = sub[qcol].to_numpy()
        m, s, n = mean_std_n(vals)
        means.append(m)
        stds.append(s)
        if penalize_failures:
            total = len(sub)
            completed = int(sub.get("completed", pd.Series(True, index=sub.index)).sum())
            annotations.append(f"N={completed}/{total}")
            # Raw mean (judge-scored, completed only) — used to draw the
            # "would-have-been" segment as a hatched extension of the bar.
            real = sub[sub["completed"].fillna(False).astype(bool)]
            raw_mean = float(real["quality_score"].dropna().mean()) if real["quality_score"].notna().any() else float("nan")
            raw_means.append(raw_mean)
        else:
            annotations.append(f"N={n}/{len(sub)}")
            raw_means.append(float("nan"))

    y_pos = np.arange(len(names))
    colors = router_palette(names)[::-1]
    rev_means = means[::-1]
    rev_stds = stds[::-1]
    rev_ann = annotations[::-1]
    rev_raw = raw_means[::-1]

    bars = ax.barh(
        y_pos, rev_means,
        xerr=rev_stds if show_errorbar else None,
        capsize=4 if show_errorbar else 0,
        color=colors, edgecolor="black", linewidth=0.5,
    )
    # Hatched extension: how much taller the bar *would have been* under the
    # raw mean (failures excluded). Makes the penalty visible per router.
    if penalize_failures:
        for i, bar in enumerate(bars):
            raw = rev_raw[i]
            if np.isnan(raw) or raw <= rev_means[i]:
                continue
            gap = raw - rev_means[i]
            ax.barh(
                bar.get_y() + bar.get_height() / 2,
                gap,
                height=bar.get_height(),
                left=bar.get_x() + bar.get_width(),
                color="none", edgecolor="black", hatch="///", linewidth=0.5,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    xlabel = "Mean Quality Score (0-10)"
    if penalize_failures:
        xlabel += "  ·  failures=0"
    ax.set_xlabel(xlabel)
    ax.set_title(title or "Overall Mean Quality by Router", fontsize=fontsize + 2)
    ax.set_xlim(0, 10.5)

    for i, bar in enumerate(bars):
        # Anchor the label past the hatched extension (if any) so it doesn't
        # overlap either the bar or the error bar whiskers.
        raw = rev_raw[i] if penalize_failures else float("nan")
        bar_end = max(rev_means[i], raw) if not np.isnan(raw) else rev_means[i]
        x_end = bar_end + (rev_stds[i] if show_errorbar else 0)
        label = f"{rev_means[i]:.2f} ({rev_ann[i]})"
        if penalize_failures and not np.isnan(raw) and raw > rev_means[i]:
            label = f"{rev_means[i]:.2f}<-{raw:.2f} ({rev_ann[i]})"
        ax.text(
            x_end + 0.2,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", fontsize=fontsize,
        )

    fig.tight_layout()
    return fig, ax


def plot_per_category_quality(
    df: pd.DataFrame,
    exclude_routers: Optional[List[str]] = None,
    router_order: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (8, 6),
    title: Optional[str] = None,
    fontsize: Optional[int] = 6,
    show_errorbar: bool = True,
    penalize_failures: bool = False,
    total_queries: Optional[int] = None,
    query_catalog: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart of mean quality per router × category.

    When `penalize_failures=True`, missing queries are padded with quality 0
    and timed-out rows are forced to 0 (judge-NaN rows still drop out).
    """
    names = filter_and_order_routers(df, exclude_routers, router_order)
    prepared, qcol = _prepare(df, penalize_failures, total_queries, query_catalog)
    cats = categories or CATEGORY_ORDER
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(cats))
    total_width = 0.8
    bar_width = total_width / max(len(names), 1)
    colors = router_palette(names)

    for i, name in enumerate(names):
        means, errs, ns = [], [], []
        for cat in cats:
            vals = prepared.loc[
                (prepared["router"] == name) & (prepared["query_category"] == cat),
                qcol,
            ].to_numpy()
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
        annotate_small_n(ax, bars, ns, threshold=3, fontsize=fontsize, y_offset_abs=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ylabel = "Mean Quality Score (0-10)"
    if penalize_failures:
        ylabel += "  ·  failures=0"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Query Category")
    ax.set_title(title or "Mean Quality by Router and Category")
    ax.set_ylim(0, 11)
    ax.legend(fontsize=fontsize + 2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax


def build_quality_summary_table(
    df: pd.DataFrame,
    *,
    total_queries: Optional[int] = None,
    query_catalog: Optional[pd.DataFrame] = None,
    router_order: Optional[List[str]] = None,
    exclude_routers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-router table: raw vs failure-penalized mean quality, with failure counts.

    Columns:
      router, total, completed, timed_out, judge_nan, dropouts,
      mean_raw (judge-scored only), mean_penalized (failures=0), penalty.
    """
    names = filter_and_order_routers(df, exclude_routers, router_order)
    padded = expand_to_all_queries(df, total_queries=total_queries, query_catalog=query_catalog)
    padded = apply_quality_penalty(padded)

    rows = []
    for name in names:
        sub = padded[padded["router"] == name]
        total = len(sub)
        completed = int(sub["completed"].sum())
        dropouts = total - completed
        timed_out = int(sub["timed_out"].fillna(False).astype(bool).sum())
        # judge_nan = completed, not timed_out, quality NaN
        judge_nan_mask = (
            sub["completed"].fillna(False).astype(bool)
            & ~sub["timed_out"].fillna(False).astype(bool)
            & sub["quality_score"].isna()
        )
        judge_nan = int(judge_nan_mask.sum())
        # Raw mean uses only rows the router actually produced (pre-padding).
        real_rows = sub[sub["completed"].fillna(False).astype(bool)]
        mean_raw = float(real_rows["quality_score"].dropna().mean()) if real_rows["quality_score"].notna().any() else float("nan")
        mean_penalized = float(sub["quality_effective"].dropna().mean()) if sub["quality_effective"].notna().any() else float("nan")
        penalty = mean_raw - mean_penalized if not (np.isnan(mean_raw) or np.isnan(mean_penalized)) else float("nan")
        rows.append({
            "router": name,
            "total": total,
            "completed": completed,
            "timed_out": timed_out - dropouts,  # timed_out on real rows only (padded rows already counted as dropouts)
            "judge_nan": judge_nan,
            "dropouts": dropouts,
            "mean_raw": mean_raw,
            "mean_penalized": mean_penalized,
            "penalty": penalty,
        })
    return pd.DataFrame(rows)
