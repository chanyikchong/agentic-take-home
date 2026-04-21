"""Shared constants for the evaluation subpackage."""

import matplotlib as mpl
import matplotlib.colors as mcolors


CATEGORY_ORDER = ["simple", "moderate", "complex", "reasoning", "coding"]

# 30-color router palette: tab20's 20 colors (fully distinct from tab10) plus
# 10 evenly-sampled colors from tab20b. `[::2]` picks every other color from
# tab20b so the added 10 span all 5 hue groups instead of clustering. tab20b
# has zero overlap with tab20, so all 30 colors are unique.
_tab30_colors = (
    list(mpl.colormaps["tab20"].colors)
    + list(mpl.colormaps["tab20b"].colors)[::2]
)
ROUTER_CMAP = mcolors.ListedColormap(_tab30_colors, name="tab30")

# MONITOR_TO_ROUTER lives in data_loading (lightweight, no matplotlib).
# Re-exported here for backward compatibility with any code that still imports
# it from constants.
from solutions.evaluation.data_loading import MONITOR_TO_ROUTER  # noqa: E402, F401

# Node wall-time segments for stacked-latency plots. `mission` and
# `latency_criticality` are inner agents inside `scoring` — including them
# alongside `scoring` would double-count, so they are excluded here.
NODE_SEGMENTS = ["meta_routing", "intent", "scoring", "decision", "fast_decision"]

# Agent segments that actually incur cost (LLM calls). `scoring` and
# `fast_decision` are wrappers with no LLM call. Used by cost node-breakdown.
COST_SEGMENTS = ["meta_routing", "intent", "mission", "latency_criticality", "decision"]

# Shared label + color map across latency and cost segments. Overlapping
# steps (meta_routing, intent, decision) keep the same colour so readers
# comparing latency/cost breakdowns see the same visual identity per step.
NODE_LABELS = {
    "meta_routing": "Meta Routing",
    "intent": "Intent",
    "scoring": "Scoring (parallel)",
    "decision": "Decision",
    "fast_decision": "Fast Decision",
    "mission": "Mission",
    "latency_criticality": "Latency Criticality",
}
_NODE_CMAP = mpl.colormaps["Set2"]
_ALL_STEPS = NODE_SEGMENTS + ["mission", "latency_criticality"]
NODE_COLORS = {seg: _NODE_CMAP(i) for i, seg in enumerate(_ALL_STEPS)}

SPLIT_COLORS = {
    "router_ms": "#a0a0a0", "inference_ms": "#4c72b0",
    "router_cost": "#a0a0a0", "inference_cost": "#4c72b0",
}
