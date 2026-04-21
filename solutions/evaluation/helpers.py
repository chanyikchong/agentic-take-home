"""Shared helpers for the evaluation subpackage.

Private (leading underscore) — not part of the public `solutions.evaluation` API.
"""

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from solutions.evaluation.constants import ROUTER_CMAP


def filter_and_order_routers(
    source: Union[pd.DataFrame, Iterable[str]],
    exclude: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
) -> List[str]:
    """Return ordered list of router names after filtering.

    Accepts a DataFrame with a `router` column or any iterable of names.
    Unordered routers are appended in their original (encountered) order.
    """
    if isinstance(source, pd.DataFrame):
        names = list(dict.fromkeys(source["router"].tolist()))
    else:
        names = list(dict.fromkeys(source))
    if exclude:
        names = [n for n in names if n not in exclude]
    if order:
        ordered = [n for n in order if n in names]
        remaining = [n for n in names if n not in ordered]
        names = ordered + remaining
    return names


def router_palette(names: List[str]) -> np.ndarray:
    """Return one color per router name from ROUTER_CMAP."""
    return ROUTER_CMAP(np.linspace(0, 1, max(len(names), 1)))


def mean_std_n(values) -> Tuple[float, float, int]:
    """Return (mean, std, n) after dropping NaN. std=0 when n<2."""
    arr = np.asarray(list(values), dtype=float) if not isinstance(values, np.ndarray) else values
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = float(np.mean(arr)) if n > 0 else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return mean, std, n


def apply_quality_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `quality_effective` column that zeroes out router failures.

    Rules:
      - If `completed` is False  → 0.0 (router never produced a response).
      - Else if `timed_out` is True → 0.0 (router produced an error/timeout).
      - Else → `quality_score` (preserves NaN; judge failures stay NaN so
        downstream aggregators drop them just like today).

    Safe to call on a frame that has not been passed through
    `expand_to_all_queries` — treats missing `completed` column as all True.
    """
    out = df.copy()
    completed = out["completed"] if "completed" in out.columns else pd.Series(True, index=out.index)
    timed_out = out["timed_out"].fillna(False).astype(bool) if "timed_out" in out.columns else pd.Series(False, index=out.index)
    failed = (~completed.astype(bool)) | timed_out
    out["quality_effective"] = np.where(failed, 0.0, out["quality_score"])
    return out


def annotate_small_n(
    ax,
    bars,
    ns: List[int],
    threshold: int = 3,
    fontsize: int = 6,
    color: str = "red",
    y_offset_frac: float = 0.02,
    y_offset_abs: float = 0.1,
) -> None:
    """Write `n=<k>` above bars whose sample size is below threshold."""
    for bar, n in zip(bars, ns):
        if 0 < n < threshold:
            h = bar.get_height()
            offset = max(h * y_offset_frac, y_offset_abs)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                f"n={n}",
                ha="center", va="bottom", fontsize=fontsize, color=color,
            )
