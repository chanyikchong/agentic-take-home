"""Unified evaluation data frames.

Loads benchmark results + monitor pickles and assembles two DataFrames
that feed all downstream analysis:

- `evaluation_df` (wide): one row per (router, query_index) with identity,
  result, router overhead totals, agent outputs, final decision, assigned
  models, and per-node error flags.
- `step_df` (long): one row per (router, query_index, step) with step
  latency and cost. Custom routers only (baselines have no monitor).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from solutions.evaluation.benchmarking import RoutingBenchmarkResult
from solutions.evaluation.data_loading import load_all_results, load_all_monitors


# Monitor step-dict keys → canonical step names used in step_df.
# "latency" (LatencyCriticality agent wall-time) renamed to avoid ambiguity
# with benchmark `latency_ms` (inference).
_STEP_RENAME = {"latency": "latency_criticality"}

# Step keys that appear in step_cost (scoring and fast_decision have no cost entry).
_COST_STEPS = ("meta_routing", "intent", "mission", "latency", "decision")


def build_evaluation_frame(
    all_results: Dict[str, List[RoutingBenchmarkResult]],
    monitors: Dict[str, dict],
    *,
    include_reasoning: bool = False,
) -> pd.DataFrame:
    """Wide frame — one row per (router, query_index).

    Args:
        all_results: Keyed by router display name → list of RoutingBenchmarkResult.
        monitors:    Keyed by router display name → monitor dict (produced by
                     `load_all_monitors`, which now handles the stem→name map
                     internally). Baselines (no monitor) are simply absent.

    Baselines (no monitor) get NaN / None / False for monitor-derived columns.
    Query source precedence: monitor `graph_states[i].query` (full) wins when
    present; falls back to `RoutingBenchmarkResult.query` (50-char truncated).
    `query_truncated` flags which source was used.
    """
    rows = []

    for router_name, results in all_results.items():
        monitor = monitors.get(router_name)
        routing_latencies = monitor["latency"] if monitor else None
        routing_costs = monitor["cost"] if monitor else None
        graph_states = monitor["graph_states"] if monitor else None

        for idx, res in enumerate(results):
            # Query source precedence: monitor full > result truncated
            if graph_states is not None and idx < len(graph_states):
                full_query = graph_states[idx].query
                query_truncated = False
            else:
                full_query = res.query
                query_truncated = res.query.endswith("...")

            tier = res.model_tier.value if hasattr(res.model_tier, "value") else str(res.model_tier)

            row = {
                # Identity
                "router": router_name,
                "query_index": idx,
                "query": full_query,
                "query_truncated": query_truncated,
                "query_category": res.query_category,
                "timed_out": res.timed_out,
                # Result
                "model_key": res.model_key,
                "deployment": res.deployment,
                "model_tier": tier,
                "inference_ms": res.latency_ms,
                "quality_score": res.quality_score,
                "inference_cost": res.cost_estimate,
                "response": res.response,
                # Router overhead totals (NaN for baselines)
                "router_ms": np.nan,
                "router_cost": np.nan,
                # Agent outputs (filled below when monitor present)
                "intent": None,
                "intent_conf": np.nan,
                "mission_score": np.nan,
                "mission_conf": np.nan,
                "latency_criticality_score": np.nan,
                "latency_criticality_conf": np.nan,
                # Final decision
                "final_model": None,
                "final_deploy": None,
                "final_conf": np.nan,
                # Assigned agent models
                "assigned_intent": None,
                "assigned_mission": None,
                "assigned_latency": None,
                "assigned_decision": None,
                # Error flags
                "err_meta": False,
                "err_intent": False,
                "err_mission": False,
                "err_latency": False,
                "err_decision": False,
                "err_count": 0,
                # Flags
                "fast_path": False,
            }
            if include_reasoning:
                row["intent_reasoning"] = None
                row["mission_reasoning"] = None
                row["latency_criticality_reasoning"] = None
                row["final_reasoning"] = None

            if monitor is not None:
                if routing_latencies is not None and idx < len(routing_latencies):
                    row["router_ms"] = float(routing_latencies[idx])
                if routing_costs is not None and idx < len(routing_costs):
                    row["router_cost"] = float(routing_costs[idx])

                if graph_states is not None and idx < len(graph_states):
                    g_state = graph_states[idx]
                    if g_state.intent is not None:
                        row["intent"] = g_state.intent.intent.value
                        row["intent_conf"] = g_state.intent.confidence
                        if include_reasoning:
                            row["intent_reasoning"] = g_state.intent.reasoning
                    if g_state.mission is not None:
                        row["mission_score"] = g_state.mission.score
                        row["mission_conf"] = g_state.mission.confidence
                        if include_reasoning:
                            row["mission_reasoning"] = g_state.mission.reasoning
                    if g_state.latency is not None:
                        row["latency_criticality_score"] = g_state.latency.score
                        row["latency_criticality_conf"] = g_state.latency.confidence
                        if include_reasoning:
                            row["latency_criticality_reasoning"] = g_state.latency.reasoning
                    if g_state.decision is not None:
                        row["final_model"] = g_state.decision.model_key
                        row["final_deploy"] = g_state.decision.deployment
                        row["final_conf"] = g_state.decision.confidence
                        if include_reasoning:
                            row["final_reasoning"] = g_state.decision.reasoning
                    assignments = g_state.model_assignments or {}
                    row["assigned_intent"] = assignments.get("intent")
                    row["assigned_mission"] = assignments.get("mission")
                    row["assigned_latency"] = assignments.get("latency")
                    row["assigned_decision"] = assignments.get("decision")
                    errors = g_state.errors or {}
                    row["err_meta"] = "meta_routing" in errors
                    row["err_intent"] = "intent" in errors
                    row["err_mission"] = "mission" in errors
                    row["err_latency"] = "latency" in errors
                    row["err_decision"] = "decision" in errors
                    row["err_count"] = sum(
                        (row["err_meta"], row["err_intent"], row["err_mission"],
                         row["err_latency"], row["err_decision"])
                    )
                    # Fast path: mission was skipped (not due to error)
                    row["fast_path"] = g_state.mission is None and not row["err_mission"]

            rows.append(row)

    return pd.DataFrame(rows)


def build_step_frame(
    all_results: Dict[str, List[RoutingBenchmarkResult]],
    monitors: Dict[str, dict],
) -> pd.DataFrame:
    """Long frame — one row per (router, query_index, step).

    Columns: router, query_index, query_category, step, latency_ms, cost.
    Only custom routers (those with a monitor) contribute rows.
    Step "latency" (LatencyCriticality agent) is renamed to
    "latency_criticality" to avoid collision with benchmark latency.
    """
    rows = []

    for router_name, results in all_results.items():
        monitor = monitors.get(router_name)
        if monitor is None:
            continue
        step_latencies = monitor["step_latency"]  # keep the latency for each node
        step_costs = monitor["step_cost"]         # keep the cost for each LLM call

        for i, r in enumerate(results):
            latency_dict = step_latencies[i] if i < len(step_latencies) else {}
            cost_dict = step_costs[i] if i < len(step_costs) else {}
            keys = set(latency_dict) | set(cost_dict)
            for key in keys:
                # rename key for latency avoid miss-use of latency in inference and routing
                step_name = _STEP_RENAME.get(key, key)
                rows.append({
                    "router": router_name,
                    "query_index": i,
                    "query_category": r.query_category,
                    "step": step_name,
                    "latency_ms": float(latency_dict.get(key, 0.0)),
                    "cost": float(cost_dict.get(key, 0.0)),
                })

    return pd.DataFrame(rows)


def evaluation_frame_from_dir(
    results_dir: Union[str, Path],
    *,
    include_reasoning: bool = False,
    filter_routers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load results + monitors and build both frames.

    Returns:
        (evaluation_df, step_df)
    """
    results = load_all_results(results_dir, filter_routers=filter_routers)
    monitors = load_all_monitors(results_dir, filter_routers=filter_routers)
    return (
        build_evaluation_frame(results, monitors, include_reasoning=include_reasoning),
        build_step_frame(results, monitors),
    )


def expand_to_all_queries(
    df: pd.DataFrame,
    *,
    total_queries: Optional[int] = None,
    query_catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pad `df` so every router has one row per query index in the canonical set.

    Semantics — three distinct states are preserved:
      - Present & completed: row kept as-is.
      - Present but timed_out: row kept as-is (caller decides how to score).
      - Missing (router never produced a result): synthetic row inserted with
        `completed=False`, `timed_out=True`, `quality_score=0.0`, category taken
        from the catalog, all other columns set to NaN/None/False.

    Args:
        df:             Output of `build_evaluation_frame`.
        total_queries:  Number of queries in the canonical benchmark. If None,
                        inferred from `query_catalog` (if given) else from
                        `df["query_index"].max() + 1`.
        query_catalog:  Optional DataFrame with columns `query_index` and
                        `query_category`. When omitted, inferred from unique
                        (index, category) pairs among present rows. Raises
                        ValueError if any padded index still lacks a category.

    Returns:
        A copy of `df` with a new `completed: bool` column and `(router ×
        total_queries)` rows in total.
    """
    df = df.copy()
    df["completed"] = True

    if query_catalog is None:
        query_catalog = (
            df[["query_index", "query_category"]]
            .dropna()
            .drop_duplicates(subset=["query_index"])
            .sort_values("query_index")
            .reset_index(drop=True)
        )
    else:
        query_catalog = query_catalog[["query_index", "query_category"]].copy()

    if total_queries is None:
        if not query_catalog.empty:
            total_queries = int(query_catalog["query_index"].max()) + 1
        else:
            total_queries = int(df["query_index"].max()) + 1 if not df.empty else 0

    category_by_index = dict(zip(query_catalog["query_index"], query_catalog["query_category"]))

    pad_rows = []
    for router in df["router"].unique():
        present = set(df.loc[df["router"] == router, "query_index"])
        for idx in range(total_queries):
            if idx in present:
                continue
            if idx not in category_by_index:
                raise ValueError(
                    f"Cannot pad router {router!r} at index {idx}: no category "
                    f"available (pass an explicit `query_catalog`)."
                )
            pad = {col: np.nan for col in df.columns}
            pad.update(
                router=router,
                query_index=idx,
                query_category=category_by_index[idx],
                timed_out=True,
                quality_score=0.0,
                completed=False,
                err_meta=False,
                err_intent=False,
                err_mission=False,
                err_latency=False,
                err_decision=False,
                err_count=0,
                fast_path=False,
                query_truncated=False,
                intent=None,
                final_model=None,
                final_deploy=None,
                assigned_intent=None,
                assigned_mission=None,
                assigned_latency=None,
                assigned_decision=None,
                response=None,
                model_key=None,
                deployment=None,
                model_tier=None,
                query=None,
            )
            pad_rows.append(pad)

    if pad_rows:
        df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)

    return df.sort_values(["router", "query_index"]).reset_index(drop=True)
