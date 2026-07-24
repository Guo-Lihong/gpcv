"""Cross-validation splitters for relational financial ML.

Three protocols, increasing in leakage-safety:

1. **naive**    : no purging (random or blocked test groups). Worst case.
2. **temporal** : Lopez de Prado label-overlap purge + embargo. The current
                  standard; oblivious to the graph.
3. **graph**    : Graph-Purged CPCV = temporal purge + embargo + purge of every
                  training node within ``L`` hops of a test node in the model's
                  adjacency. Enforces the leakage horizon R(L) = h + L*delta + e.

The graph term is what temporal-only purging misses; see docs/THEOREM.md.
"""
from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# Graph reachability                                                           #
# --------------------------------------------------------------------------- #
def l_hop_reach(adj: sp.spmatrix, seed_mask: np.ndarray, L: int) -> np.ndarray:
    """Boolean mask of all nodes within ``L`` hops of any seed node (undirected).

    Implements the receptive-field set N_L(seeds). For a symmetric 0/1 adjacency
    this equals the support of (A + I)^L restricted to the seed columns.
    """
    reached = np.asarray(seed_mask, dtype=bool).copy()
    if L <= 0:
        return reached
    A = adj.tocsr()
    for _ in range(int(L)):
        # one hop: neighbours of the current frontier
        nbr = A.dot(reached.astype(np.float32)) > 0
        new = reached | np.asarray(nbr).ravel()
        if new.sum() == reached.sum():
            break  # converged early
        reached = new
    return reached


# --------------------------------------------------------------------------- #
# Purge masks                                                                  #
# --------------------------------------------------------------------------- #
def temporal_purge_mask(
    node_time: np.ndarray,
    test_mask: np.ndarray,
    label_horizon: int,
    embargo: int,
) -> np.ndarray:
    """Nodes to REMOVE from training due to temporal label-overlap + embargo.

    A training node at time ``t`` carries a label spanning ``(t, t+h]``. It is
    purged if that window, widened by the embargo, intersects the test period's
    information window ``[t_lo - e, t_hi + h + e]``.
    """
    test_mask = np.asarray(test_mask, dtype=bool)
    if not test_mask.any():
        return np.zeros_like(test_mask)
    t_lo = int(node_time[test_mask].min())
    t_hi = int(node_time[test_mask].max())
    h, e = int(label_horizon), int(embargo)
    lo = t_lo - h - e
    hi = t_hi + h + e
    bad = (node_time >= lo) & (node_time <= hi)
    return bad & ~test_mask


def graph_purge_mask(
    adj: sp.spmatrix,
    test_mask: np.ndarray,
    L: int,
) -> np.ndarray:
    """Nodes to REMOVE from training because a test node lies within their
    L-hop receptive field in ``adj`` (the graph the model actually message-passes
    over). This is the graph term the temporal purge misses.
    """
    test_mask = np.asarray(test_mask, dtype=bool)
    reached = l_hop_reach(adj, test_mask, L)
    return reached & ~test_mask


def train_mask_for_protocol(
    protocol: str,
    node_time: np.ndarray,
    test_mask: np.ndarray,
    adj: sp.spmatrix,
    L: int,
    label_horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return the boolean TRAIN mask for a given protocol and test fold.

    protocol in {"naive", "temporal", "graph"}.
    For "graph", pass the *model's* adjacency (use adj_pit to additionally close
    the edge-construction channel — then the L-hop term purges nothing extra).
    """
    test_mask = np.asarray(test_mask, dtype=bool)
    train = ~test_mask
    if protocol == "naive":
        return train
    bad = temporal_purge_mask(node_time, test_mask, label_horizon, embargo)
    if protocol == "temporal":
        return train & ~bad
    if protocol == "graph":
        bad = bad | graph_purge_mask(adj, test_mask, L)
        return train & ~bad
    raise ValueError("protocol must be one of {'naive','temporal','graph'}")


# --------------------------------------------------------------------------- #
# Combinatorial Purged CV over time blocks                                     #
# --------------------------------------------------------------------------- #
def cpcv_time_blocks(
    node_time: np.ndarray,
    n_blocks: int = 6,
    n_test_blocks: int = 2,
) -> Iterator[np.ndarray]:
    """Yield test masks for each CPCV combination of contiguous time blocks.

    Time is partitioned into ``n_blocks`` contiguous ranges; every combination of
    ``n_test_blocks`` blocks is a test group (the remaining blocks form the pool
    from which the protocol then purges). This reconstructs the standard
    combinatorial purged backtest paths.
    """
    node_time = np.asarray(node_time)
    t_min, t_max = int(node_time.min()), int(node_time.max())
    edges = np.linspace(t_min, t_max + 1, n_blocks + 1).astype(int)
    block_of = np.searchsorted(edges, node_time, side="right") - 1
    block_of = np.clip(block_of, 0, n_blocks - 1)
    for combo in combinations(range(n_blocks), n_test_blocks):
        yield np.isin(block_of, combo)


def make_folds(
    protocol: str,
    node_time: np.ndarray,
    adj: sp.spmatrix,
    L: int,
    label_horizon: int = 1,
    embargo: int = 1,
    n_blocks: int = 6,
    n_test_blocks: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialize (train_mask, test_mask) folds for a protocol over CPCV blocks."""
    folds = []
    for test_mask in cpcv_time_blocks(node_time, n_blocks, n_test_blocks):
        train_mask = train_mask_for_protocol(
            protocol, node_time, test_mask, adj, L, label_horizon, embargo
        )
        folds.append((train_mask, test_mask))
    return folds
