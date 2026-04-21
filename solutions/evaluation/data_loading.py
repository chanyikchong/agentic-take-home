import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from solutions.evaluation.benchmarking import RoutingBenchmarkResult


# Maps monitor pickle file stem (e.g. "meta_router_v1") to the router name
# used as a key in `all_results` (e.g. "MetaRouterV1"). Kept here rather than
# in constants.py so `load_all_monitors` doesn't pull matplotlib into the
# loader call chain.
MONITOR_TO_ROUTER: Dict[str, str] = {
    "meta_router_v1": "MetaRouterV1",
    "meta_router_v2": "MetaRouterV2",
    "meta_router_v3": "MetaRouterV3",
    "meta_router_v4": "MetaRouterV4",
    "meta_router_v5": "MetaRouterV5",
    "graph_router_v1": "GraphRouterV1",
    "graph_router_v2": "GraphRouterV2",
    "graph_router_v3": "GraphRouterV3",
    "graph_router_v4": "GraphRouterV4",
    "graph_router_v5": "GraphRouterV5",
    "fast_graph_router_v1": "FastGraphRouterV1",
    "fast_graph_router_v2": "FastGraphRouterV2",
    "fast_graph_router_v3": "FastGraphRouterV3",
    "fast_graph_router_v4": "FastGraphRouterV4",
    "fast_graph_router_v5": "FastGraphRouterV5",
}


def load_all_results(
    results_dir: Union[str, Path],
    filter_routers: Optional[List[str]] = None,
) -> Dict[str, List[RoutingBenchmarkResult]]:
    """Load all benchmark results from pickle files into a single dict.

    Keyed by router display name (e.g. `"MetaRouterV1"`). Raises on duplicate
    router names across files.
    """
    results_dir = Path(results_dir)
    all_results: Dict[str, List[RoutingBenchmarkResult]] = {}

    # Load baselines
    baseline_path = results_dir / "baseline_results.pkl"
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            baseline_data: dict = pickle.load(f)
        all_results.update(baseline_data)

    # Load custom router results
    for pkl_path in sorted(results_dir.glob("*_result.pkl")):
        if pkl_path.name == "baseline_results.pkl":
            continue
        with open(pkl_path, "rb") as f:
            router_data: dict = pickle.load(f)
        for router_name, results in router_data.items():
            if router_name in all_results:
                raise ValueError(
                    f"Duplicate router name '{router_name}' found in "
                    f"{pkl_path.name} and a previously loaded file."
                )
            if not filter_routers or router_name in filter_routers:
                all_results[router_name] = results

    return all_results


def load_all_monitors(
    results_dir: Union[str, Path],
    name_map: Optional[Mapping[str, str]] = None,
    filter_routers: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Load all monitor pickle files, keyed by router display name.
    """
    results_dir = Path(results_dir)
    mapping = MONITOR_TO_ROUTER if name_map is None else name_map
    monitors: Dict[str, dict] = {}

    for pkl_path in sorted(results_dir.glob("*_monitor.pkl")):
        # Derive stem: "fast_graph_router_v1_monitor.pkl" -> "fast_graph_router_v1"
        stem = pkl_path.stem.replace("_monitor", "")
        router_name = mapping.get(stem)
        if router_name is None:
            warnings.warn(
                f"Skipping monitor file {pkl_path.name!r}: stem {stem!r} "
                f"not present in name_map.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if router_name in monitors:
            raise ValueError(
                f"Duplicate router name '{router_name}' resolved from "
                f"{pkl_path.name} and a previously loaded monitor file."
            )
        if not filter_routers or router_name in filter_routers:
            with open(pkl_path, "rb") as f:
                monitors[router_name] = pickle.load(f)

    return monitors
