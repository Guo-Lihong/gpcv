"""Model-free leakage diagnostics for temporal / relational graphs.

Nothing here trains a model. Given only an adjacency and per-node timestamps these
functions measure the three quantities the leakage horizon R(L) = h + L*delta + embargo
is built from, and the two channels through which message passing violates it:

* :func:`edge_temporal_reach` -- the **measured** edge temporal reach ``delta``.
  R(L) is stated in terms of a delta that is usually *assumed*. On a real graph delta
  is an empirical quantity: the distribution of |t_i - t_j| over edges. Its maximum is
  the delta that appears in the bound; its upper quantiles say whether that maximum is
  representative or a tail artefact.

* :func:`future_reach_fraction` -- the **inference channel**. Fraction of nodes whose
  L-hop receptive field contains a strictly later timestamp. Any such node is predicted
  using information dated after its own decision time, regardless of how the train/test
  split was drawn. This is exactly what an undirected graph re-admits at L >= 2 even
  when every edge is point-in-time, and what the directed operator drives to zero.

* :func:`contamination_curve` -- the **training channel**. Number of *training* nodes
  that still reach a test node in L hops after the temporal purge has been applied.
  This is the residue the L-hop graph purge exists to remove, as a function of depth.

Conventions
-----------
Adjacencies follow the library convention ``A[target, source]``: information flows from
column to row. For a symmetric (undirected) adjacency the distinction is vacuous. All
reach computations therefore follow rows -> columns, i.e. they answer "whose information
arrives at node i", which is what a message-passing layer actually does.

Reach sets are "within L hops" and include the node itself (consistent with
:func:`gpcv.splitters.l_hop_reach` and with a GNN's self-loop).
"""
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .splitters import l_hop_reach, temporal_purge_mask

__all__ = [
    "edge_temporal_reach",
    "future_reach_fraction",
    "contamination_curve",
    "max_reach_time",
]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _row_max(A: sp.csr_matrix, vals: np.ndarray) -> np.ndarray:
    """``out[i] = max(vals[j] for j in A[i].indices)``; -inf for empty rows."""
    n = A.shape[0]
    out = np.full(n, -np.inf)
    if A.nnz == 0:
        return out
    nonempty = np.diff(A.indptr) > 0
    starts = A.indptr[:-1][nonempty]
    out[nonempty] = np.maximum.reduceat(vals[A.indices], starts)
    return out


def max_reach_time(adj: sp.spmatrix, node_time: np.ndarray, L: int) -> np.ndarray:
    """Newest timestamp reachable within ``L`` hops (self included).

    ``out[i] = max{ node_time[j] : j reachable from i in <= L hops following
    A[target, source] }``. Always >= ``node_time[i]``.
    """
    A = adj.tocsr()
    t = np.asarray(node_time, dtype=np.float64)
    best = t.copy()
    for _ in range(int(L)):
        nxt = _row_max(A, best)
        new = np.maximum(best, nxt)
        if np.array_equal(new, best):
            break
        best = new
    return best


# --------------------------------------------------------------------------- #
# 1. measured edge temporal reach (delta)                                      #
# --------------------------------------------------------------------------- #
def edge_temporal_reach(
    adj: sp.spmatrix,
    node_time: np.ndarray,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99, 1.0),
    histogram: bool = False,
    hist_max: int = 50,
) -> Dict:
    """Empirical distribution of the time gap bridged by a single edge.

    Parameters
    ----------
    adj : sparse ``A[target, source]``.
    node_time : (M,) per-node timestamps.
    quantiles : quantiles of ``|t_target - t_source|`` to report.
    histogram : if True also return the full integer histogram of the signed lag
        ``t_target - t_source`` clipped at ``+-hist_max`` (key ``"lag_hist"``,
        a ``{lag: count}`` dict; the clipped bins are reported as ``<=-hist_max``
        and ``>=hist_max``).

    Returns
    -------
    dict with ``n_edges``; ``delta_max`` (the delta of R(L) = h + L*delta + e),
    ``delta_mean``, ``delta_q`` (quantiles of |lag|); the signed-lag summary
    ``lag_min``/``lag_max``/``lag_mean``; the fractions of edges that are
    ``frac_backward`` (source strictly earlier -- time-respecting),
    ``frac_contemporaneous`` (same timestamp) and ``frac_forward`` (source strictly
    later -- the edge itself looks ahead); and ``is_symmetric``.
    """
    A = adj.tocoo()
    t = np.asarray(node_time)
    if A.nnz == 0:
        return dict(n_edges=0, delta_max=0, delta_mean=float("nan"),
                    delta_q={str(q): float("nan") for q in quantiles},
                    lag_min=0, lag_max=0, lag_mean=float("nan"),
                    frac_backward=float("nan"), frac_contemporaneous=float("nan"),
                    frac_forward=float("nan"), is_symmetric=True)
    lag = t[A.row].astype(np.int64) - t[A.col].astype(np.int64)   # target - source
    ab = np.abs(lag)
    d = dict(
        n_edges=int(A.nnz),
        delta_max=int(ab.max()),
        delta_mean=float(ab.mean()),
        delta_q={str(q): float(np.quantile(ab, q)) for q in quantiles},
        lag_min=int(lag.min()),
        lag_max=int(lag.max()),
        lag_mean=float(lag.mean()),
        frac_backward=float((lag > 0).mean()),
        frac_contemporaneous=float((lag == 0).mean()),
        frac_forward=float((lag < 0).mean()),
        is_symmetric=bool((abs(adj.tocsr() - adj.tocsr().T)).nnz == 0),
    )
    if histogram:
        clipped = np.clip(lag, -hist_max, hist_max)
        vals, cnts = np.unique(clipped, return_counts=True)
        hist = {}
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if v == -hist_max:
                key = f"<=-{hist_max}"
            elif v == hist_max:
                key = f">={hist_max}"
            else:
                key = str(int(v))
            hist[key] = int(c)
        d["lag_hist"] = hist
    return d


# --------------------------------------------------------------------------- #
# 2. inference channel: does the receptive field contain the future?           #
# --------------------------------------------------------------------------- #
def future_reach_fraction(
    adj: sp.spmatrix,
    node_time: np.ndarray,
    L: int,
    node_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Fraction of nodes whose ``L``-hop receptive field contains a later timestamp.

    A node with ``max_reach_time > node_time`` is predicted from data dated after its
    own decision time -- look-ahead in the *inference* path, independent of any
    train/test split. Under a strictly directed past-only operator this is 0 at every
    depth; under an undirected graph it becomes large from L >= 2 even when every edge
    is point-in-time.

    ``node_mask`` restricts the population the fraction is computed over (the reach
    itself is always computed on the full graph).

    Returns dict with ``L``, ``n_nodes`` (population size), ``n_future``,
    ``frac_future``, and the look-ahead magnitude ``lead_mean``/``lead_max``
    (``max_reach_time - node_time``, over the whole population).
    """
    t = np.asarray(node_time).astype(np.int64)
    best = max_reach_time(adj, t, L)
    lead = best - t
    m = np.ones(t.shape[0], bool) if node_mask is None else np.asarray(node_mask, bool)
    lead_m = lead[m]
    n = int(m.sum())
    return dict(
        L=int(L),
        n_nodes=n,
        n_future=int((lead_m > 0).sum()),
        frac_future=float((lead_m > 0).mean()) if n else float("nan"),
        lead_mean=float(lead_m.mean()) if n else float("nan"),
        lead_max=int(lead_m.max()) if n else 0,
    )


# --------------------------------------------------------------------------- #
# 3. training channel: residual contamination vs depth                         #
# --------------------------------------------------------------------------- #
def contamination_curve(
    adj: sp.spmatrix,
    node_time: np.ndarray,
    test_mask: np.ndarray,
    L_max: int,
    h: int = 1,
    embargo: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> list:
    """Residual contaminated TRAINING nodes as a function of depth ``L``.

    For each ``L = 1..L_max``: build the training set under the temporal-only protocol
    (everything not in test, minus the label-overlap + embargo purge), then count how
    many of those training nodes still have a test node inside their ``L``-hop receptive
    field. Those are the nodes the L-hop graph purge removes -- and the ones a
    temporal-only purge silently keeps.

    ``valid_mask`` (e.g. ``meta["valid"]``) restricts the node population.

    Returns a list of dicts, one per ``L``, with ``n_train_temporal``,
    ``n_contaminated``, ``frac_contaminated`` (of the temporal training set),
    ``n_train_graph`` (after also applying the L-hop purge),
    ``n_removed_by_graph_purge`` (== ``n_contaminated``),
    ``frac_removed_by_graph_purge``, ``n_contaminated_naive`` (same count with no
    temporal purge at all, i.e. the raw size of the L-hop halo), and
    ``margin_days`` = ``t_lo - min(node_time)`` over the contaminated nodes (how far
    before the test boundary the contamination reaches; NaN if none).
    """
    test = np.asarray(test_mask, bool)
    t = np.asarray(node_time)
    valid = np.ones(t.shape[0], bool) if valid_mask is None else np.asarray(valid_mask, bool)
    bad_t = temporal_purge_mask(t, test, h, embargo)
    train_temporal = valid & ~test & ~bad_t
    train_naive = valid & ~test
    t_lo = int(t[test].min()) if test.any() else 0
    rows = []
    for L in range(1, int(L_max) + 1):
        reach = l_hop_reach(adj, test, L) & ~test
        contam = train_temporal & reach
        n_c = int(contam.sum())
        n_tr = int(train_temporal.sum())
        rows.append(dict(
            L=int(L),
            n_train_temporal=n_tr,
            n_contaminated=n_c,
            frac_contaminated=float(n_c / n_tr) if n_tr else float("nan"),
            n_train_graph=int((train_temporal & ~reach).sum()),
            n_removed_by_graph_purge=n_c,
            frac_removed_by_graph_purge=float(n_c / n_tr) if n_tr else float("nan"),
            n_contaminated_naive=int((train_naive & reach).sum()),
            margin_days=(int(t_lo - t[contam].min()) if n_c else float("nan")),
        ))
    return rows
