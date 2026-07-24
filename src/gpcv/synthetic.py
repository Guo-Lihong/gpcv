"""Synthetic temporal relational graphs for studying message-passing leakage.

We generate a graph over (asset, time) *samples* with a controllable **edge
temporal reach** ``delta`` (the largest time gap a single edge bridges). This lets
us exhibit — and unit-test with known ground truth — the depth-dependent leakage
horizon  R(L) = h + L*delta  from docs/THEOREM.md.

Two adjacencies are produced from the same peer structure:

* ``adj``      : the graph actually used at train time. If ``point_in_time`` is
                 False (the *leaky*/transductive setting) an edge from a node at
                 time ``t`` may connect to a peer at time in ``[t-delta, t+delta]``
                 — so edges straddle any train/test boundary.
* ``adj_pit``  : the strictly point-in-time reconstruction, where an edge from a
                 node at ``t`` only reaches peers at time in ``[t-delta, t]``. This
                 is what the Graph-Purged CPCV *fix* enforces; on the forward axis
                 delta collapses and the graph leakage term vanishes.

The label model is a low-rank factor structure with i.i.d.-in-time factors, so
returns are **cross-sectionally correlated but not predictable from the past** —
i.e. the true out-of-sample skill of a leakage-free model is ~0, and any positive
out-of-sample IC is attributable to leakage.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class TemporalGraph:
    node_time: np.ndarray            # (M,) int  decision time of each node
    asset_id: np.ndarray             # (M,) int  asset index of each node
    adj: sp.csr_matrix               # (M, M) symmetric 0/1 — train-time (leaky) undirected graph
    adj_pit: sp.csr_matrix           # (M, M) symmetric 0/1 — PIT edges but UNDIRECTED (still leaks at L>=2)
    adj_pit_dir: sp.csr_matrix       # (M, M) DIRECTED strictly-past [target, source] — the strict fix
    features: Optional[np.ndarray] = None   # (M, F)
    adj_pit_contemp: Optional[sp.csr_matrix] = None  # DIRECTED t'<=t (keeps same-day, forbids future)
    labels: Optional[np.ndarray] = None     # (M,)
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_time.shape[0])

    def node_id(self, asset: int, t: int) -> int:
        """Map (asset, time) -> flat node index (row-major over time)."""
        return int(t) * int(self.meta["n_assets"]) + int(asset)


def _symmetrize(rows, cols, n):
    """Build a symmetric 0/1 CSR adjacency (self loops filtered, deduplicated)."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    keep = rows != cols
    r = np.concatenate([rows[keep], cols[keep]])
    c = np.concatenate([cols[keep], rows[keep]])
    data = np.ones(r.shape[0], dtype=np.int8)
    A = sp.coo_matrix((data, (r, c)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1  # collapse multiplicities to 0/1
    return A


def make_temporal_graph(
    n_assets: int = 60,
    n_times: int = 120,
    k_neighbors: int = 8,
    delta: int = 5,
    n_factors: int = 3,
    label_horizon: int = 1,
    n_features: int = 8,
    idio: float = 1.0,
    feat_noise: float = 1.0,
    signal_features: int = 0,
    seed: int = 0,
) -> TemporalGraph:
    """Generate a synthetic temporal relational graph.

    Parameters
    ----------
    n_assets, n_times : grid size; number of nodes M = n_assets * n_times.
    k_neighbors : out-degree before symmetrization (peers per node).
    delta : edge temporal reach; a single edge bridges at most this many steps.
    n_factors : rank of the common-factor return structure.
    label_horizon : h; label of node at t depends on window (t, t+h].
    n_features : dimension of (mostly uninformative) node features.
    idio : idiosyncratic noise scale in returns.
    signal_features : if >0, that many features are *legitimately* predictive
        (past-only), giving a small non-zero true skill; default 0 = pure null.
    seed : RNG seed.
    """
    rng = np.random.default_rng(seed)
    N, T = n_assets, n_times
    M = N * T

    # ---- node index bookkeeping (row-major over time: node = t*N + i) ----
    tt, ii = np.divmod(np.arange(M), N)
    node_time = tt.astype(np.int64)
    asset_id = ii.astype(np.int64)

    # ---- factor return model: cross-sectionally correlated, time-unpredictable ----
    # Returns share common factors (so peers co-move -> graph label-leakage is
    # exploitable), but the factors are i.i.d. in time, so the forward return is
    # NOT predictable from the past. True leak-free skill therefore comes only from
    # the optional 'announced' signal below.
    B = rng.standard_normal((N, n_factors))                      # asset loadings
    f = rng.standard_normal((T + label_horizon + 1, n_factors))  # i.i.d.-in-time factors
    factor = np.zeros((N, T))                                     # future-factor driver (no idio)
    for t in range(T):
        fac = f[t + 1: t + 1 + label_horizon].sum(axis=0)        # FUTURE factors only
        factor[:, t] = B @ fac
    fwd = factor + idio * rng.standard_normal((N, T))

    # 'announced' signal a_s: a scalar known at decision time t (leak-free predictor).
    # signal_features is used as a strength knob: 0 => no legit predictor from own feats.
    a = rng.standard_normal(M)
    signal_strength = 0.5 * float(signal_features)

    # labels indexed by node = t*N + i  ->  fwd[i, t] (+ predictable part)
    factor_node = factor[asset_id, node_time]
    labels = fwd[asset_id, node_time] + signal_strength * a

    # ---- node features ----
    # feature 0: leak-free predictable signal a_s (only informative if signal_strength>0)
    # feature 1: a NOISY view of the node's own FUTURE-factor driver. Using it for the
    #   node itself would be look-ahead; the leakage experiments therefore predict a node
    #   from its NEIGHBOURS only (self excluded). Under a strictly point-in-time graph a
    #   node's neighbours are all in the past, so their future-factor views are
    #   uninformative about its own future label => true skill ~ 0. Under a leaky graph,
    #   contemporaneous / cross-boundary neighbours leak the future factor.
    X = rng.standard_normal((M, n_features)).astype(np.float64)
    X[:, 0] = a
    X[:, 1] = factor_node + feat_noise * rng.standard_normal(M)

    # ---- peer structure: connect assets with similar factor loadings ----
    # cosine similarity of loadings -> each asset's top peers (co-moving cluster)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = Bn @ Bn.T
    np.fill_diagonal(sim, -np.inf)
    peer_rank = np.argsort(-sim, axis=1)  # (N, N) peers by similarity

    # ---- build edges over samples with temporal reach delta ----
    rows, cols = [], []
    rows_pit, cols_pit = [], []
    lo_leaky, hi_leaky = -delta, delta
    for node in range(M):
        i, t = asset_id[node], node_time[node]
        peers = peer_rank[i, :k_neighbors]
        for j in peers:
            # leaky edge: peer sample at t' in [t-delta, t+delta]
            tp = t + rng.integers(lo_leaky, hi_leaky + 1)
            if 0 <= tp < T:
                rows.append(node); cols.append(tp * N + j)
            # point-in-time edge: peer sample at t' in [t-delta, t-1] (strictly past)
            tq = t - rng.integers(1, delta + 1)
            if 0 <= tq < T:
                rows_pit.append(node); cols_pit.append(tq * N + j)

    adj = _symmetrize(np.array(rows), np.array(cols), M)
    adj_pit = _symmetrize(np.array(rows_pit), np.array(cols_pit), M)
    # directed past-only operator: entry [target, source] with source strictly earlier.
    rp = np.asarray(rows_pit); cp = np.asarray(cols_pit)
    keep = rp != cp
    adj_pit_dir = sp.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (rp[keep], cp[keep])), shape=(M, M)
    ).tocsr()
    adj_pit_dir.sum_duplicates()
    adj_pit_dir.data[:] = 1

    return TemporalGraph(
        node_time=node_time,
        asset_id=asset_id,
        adj=adj,
        adj_pit=adj_pit,
        adj_pit_dir=adj_pit_dir,
        features=X,
        labels=labels,
        meta=dict(n_assets=N, n_times=T, k_neighbors=k_neighbors, delta=delta,
                  n_factors=n_factors, label_horizon=label_horizon, seed=seed,
                  signal_features=signal_features),
    )
