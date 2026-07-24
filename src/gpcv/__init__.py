"""gpcv — Graph-Purged Cross-Validation.

A standalone, permissively-licensed, dataset-agnostic toolkit for detecting and correcting
*message-passing leakage* in relational financial machine learning. It is API-compatible with
the temporal purged/embargoed CV libraries purgedcv / skfolio but does not depend on or wrap
them (the temporal purge/embargo/CPCV blocks are implemented here from scratch).

Core idea (see docs/THEOREM.md): for an L-layer message-passing GNN whose edges
bridge a time gap of at most delta, the leakage horizon around a train/test
boundary is R(L) = h + L*delta + embargo. Temporal-only purging removes only the
label-horizon term h; the graph term L*delta grows with depth and must be purged
across the L-hop neighborhood.
"""
from .synthetic import TemporalGraph, make_temporal_graph
from .splitters import (
    l_hop_reach,
    temporal_purge_mask,
    graph_purge_mask,
    cpcv_time_blocks,
    make_folds,
)
from .diagnostics import (
    edge_temporal_reach,
    future_reach_fraction,
    contamination_curve,
    max_reach_time,
)

__all__ = [
    "TemporalGraph",
    "make_temporal_graph",
    "l_hop_reach",
    "temporal_purge_mask",
    "graph_purge_mask",
    "cpcv_time_blocks",
    "make_folds",
    "edge_temporal_reach",
    "future_reach_fraction",
    "contamination_curve",
    "max_reach_time",
]

__version__ = "0.1.0"
