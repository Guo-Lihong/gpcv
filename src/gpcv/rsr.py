"""Load the RSR relational-stock data into a `TemporalGraph`.

The RSR relation graph (sector/wiki) is *within-day* cross-sectional. To study the
depth-dependent TEMPORAL leakage horizon on real data we lift it to a spatiotemporal
graph over (stock, day) nodes: node (i,t) connects to relation- or correlation-peers
(j, t') with t' within an edge temporal reach delta. This instantiates exactly the
`TemporalGraph` abstraction used by the synthetic study, so the same splitters,
models and metrics apply.

Edge sources (the edge-construction channel):
  * 'relation'  : static sector+wiki peers (external; minimal construction leakage).
  * 'corr_full' : correlation peers from the FULL sample (leaky — includes test span).
  * 'corr_asof' : correlation peers from a trailing window ending before t (point-in-time).

Data: Feng et al., "Temporal Relational Ranking for Stock Prediction" (TOIS 2019).
NOTE: this dataset is AGPL-3.0 and is NOT redistributed with the gpcv library.
"""
import os

import numpy as np
import scipy.sparse as sp

from .synthetic import TemporalGraph

MISS = -1234.0  # missing-value sentinel in the RSR CSVs


def load_tickers(data_dir, market):
    path = os.path.join(data_dir, f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            tok = ln.replace("\t", ",").split(",")[0].strip()
            if tok:
                out.append(tok)
    return out


def load_eod(data_dir, market, tickers, steps=1):
    """Return eod[N,T,F], gt[N,T] (return over `steps` days), mask[N,T], close[N,T]."""
    price_dir = os.path.join(data_dir, "2013-01-01")
    eod = gt = mask = close = None
    for idx, tk in enumerate(tickers):
        s = np.genfromtxt(os.path.join(price_dir, f"{market}_{tk}_1.csv"),
                          dtype=np.float32, delimiter=",")
        if market == "NASDAQ":
            s = s[:-1, :]                      # RSR drops NASDAQ's last (dirty) day
        if eod is None:
            T, C = s.shape
            eod = np.zeros((len(tickers), T, C - 1), np.float32)
            gt = np.zeros((len(tickers), T), np.float32)
            mask = np.ones((len(tickers), T), np.float32)
            close = np.zeros((len(tickers), T), np.float32)
        px = s[:, -1]
        miss = np.abs(px - MISS) < 1e-6
        mask[idx][miss] = 0.0
        valid_prev = np.zeros(s.shape[0], bool)
        valid_prev[steps:] = ~ (np.abs(px[:-steps] - MISS) < 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.zeros(s.shape[0], np.float32)
            r[steps:] = (px[steps:] - px[:-steps]) / px[:-steps]
        good = (~miss) & valid_prev
        gt[idx][good] = r[good]
        feats = s[:, 1:].copy()
        feats[np.abs(feats - MISS) < 1e-6] = 1.1   # RSR's missing-fill
        eod[idx] = feats
        close[idx] = px
    return eod, gt, mask, close


def _relation_adj(data_dir, market, n_all):
    """Union sector+wiki adjacency (binary, N_all x N_all) among all qualified tickers."""
    A = np.zeros((n_all, n_all), np.float32)
    for sub, fn in (("sector_industry", f"{market}_industry_relation.npy"),
                    ("wikidata", f"{market}_wiki_relation.npy")):
        p = os.path.join(data_dir, "relation", sub, fn)
        if not os.path.exists(p):
            continue
        rel = np.load(p, mmap_mode="r")           # [N,N,R]
        # collapse relation types -> binary adjacency, in blocks to bound memory
        for i0 in range(0, rel.shape[0], 256):
            blk = np.asarray(rel[i0:i0 + 256])    # materialise a row-block
            A[i0:i0 + 256] += (blk.sum(axis=2) > 0).astype(np.float32)
            del blk
        del rel
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(np.int8)


def _topk_peers(adj_or_sim, k, rng):
    """For each row, up to k neighbour indices (by similarity, or random among adj)."""
    N = adj_or_sim.shape[0]
    peers = []
    for i in range(N):
        row = adj_or_sim[i].astype(float).copy()
        row[i] = -np.inf
        nz = np.where(np.isfinite(row) & (row > 0))[0] if adj_or_sim.dtype == np.int8 \
            else np.argsort(-row)[:k]
        if adj_or_sim.dtype == np.int8:
            if nz.size > k:
                nz = rng.choice(nz, k, replace=False)
            peers.append(nz)
        else:
            peers.append(nz[:k])
    return peers


def build_rsr_temporal_graph(
    data_dir, market="NASDAQ", n_assets=200, day_start=0, n_times=500,
    k_neighbors=8, delta=3, edge_source="relation", corr_window=60, seed=0,
):
    """Build a spatiotemporal `TemporalGraph` from RSR data.

    Nodes = (asset, day) for the first ``n_assets`` tickers over ``n_times`` days
    starting at ``day_start``. Label = next-day return; features = day-t EOD features.
    """
    rng = np.random.default_rng(seed)
    tickers_all = load_tickers(data_dir, market)
    price_dir = os.path.join(data_dir, "2013-01-01")
    # keep the first n_assets tickers that actually have a price file; remember their
    # positions in tickers_all so the relation matrix can be sliced consistently.
    tickers, kept_idx = [], []
    for gi, tk in enumerate(tickers_all):
        if len(tickers) >= n_assets:
            break
        if os.path.exists(os.path.join(price_dir, f"{market}_{tk}_1.csv")):
            tickers.append(tk); kept_idx.append(gi)
    kept_idx = np.array(kept_idx)
    eod, gt, mask, close = load_eod(data_dir, market, tickers)
    N = len(tickers)
    Tmax = eod.shape[1]
    day_end = min(day_start + n_times, Tmax - 1)     # -1: need t+1 for the label
    days = np.arange(day_start, day_end)
    T = len(days)

    # ---- peers ----
    if edge_source == "relation":
        radj_all = _relation_adj(data_dir, market, len(tickers_all))
        radj = radj_all[np.ix_(kept_idx, kept_idx)]
        peers = _topk_peers(radj, k_neighbors, rng)
    else:
        rets = np.diff(close[:N, days[0]:days[-1] + 2], axis=1)  # [N, T]
        rets = np.nan_to_num(rets)
        if edge_source == "corr_full":
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = np.nan_to_num(np.corrcoef(rets))
            peers = _topk_peers(sim.astype(np.float64), k_neighbors, rng)
        elif edge_source == "corr_asof":
            peers = None  # computed per-day below
        else:
            raise ValueError(edge_source)

    # ---- flat node bookkeeping: node = ti*N + i (ti indexes `days`) ----
    M = N * T
    node_time = np.repeat(np.arange(T), N).astype(np.int64)     # 0..T-1 (day index)
    asset_id = np.tile(np.arange(N), T).astype(np.int64)
    features = eod[asset_id, days[node_time]]                   # [M, F]
    labels = gt[asset_id, days[node_time] + 1]                  # next-day return
    valid = (mask[asset_id, days[node_time]] > 0.5) & (mask[asset_id, days[node_time] + 1] > 0.5)

    # ---- edges over samples with temporal reach delta ----
    rows, cols, rows_pit, cols_pit = [], [], [], []
    rows_ct, cols_ct = [], []   # contemporaneous PIT: t' <= t (keeps same-day, forbids future)
    for ti in range(T):
        t = days[ti]
        if edge_source == "corr_asof":
            lo = max(days[0], t - corr_window)
            seg = np.diff(close[:N, lo:t + 1], axis=1)
            seg = np.nan_to_num(seg)
            if seg.shape[1] >= 5:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sim = np.nan_to_num(np.corrcoef(seg))
                day_peers = _topk_peers(sim.astype(np.float64), k_neighbors, rng)
            else:
                day_peers = [np.array([], int)] * N
        else:
            day_peers = peers
        base = ti * N
        for i in range(N):
            for j in day_peers[i]:
                # leaky: peer at t' in [ti-delta, ti+delta]
                tj = ti + int(rng.integers(-delta, delta + 1))
                if 0 <= tj < T:
                    rows.append(base + i); cols.append(tj * N + j)
                # PIT (directed, strictly earlier)
                tq = ti - int(rng.integers(1, delta + 1))
                if 0 <= tq < T:
                    rows_pit.append(base + i); cols_pit.append(tq * N + j)
                # contemporaneous PIT (directed, t' <= t: same-day allowed, no future)
                tc = ti - int(rng.integers(0, delta + 1))
                if 0 <= tc < T:
                    rows_ct.append(base + i); cols_ct.append(tc * N + j)

    def _sym(r, c):
        r = np.asarray(r); c = np.asarray(c); keep = r != c
        rr = np.concatenate([r[keep], c[keep]]); cc = np.concatenate([c[keep], r[keep]])
        A = sp.coo_matrix((np.ones(rr.size, np.int8), (rr, cc)), shape=(M, M)).tocsr()
        A.sum_duplicates(); A.data[:] = 1
        return A

    adj = _sym(rows, cols)
    adj_pit = _sym(rows_pit, cols_pit)
    rp = np.asarray(rows_pit); cp = np.asarray(cols_pit); keep = rp != cp
    adj_pit_dir = sp.coo_matrix((np.ones(int(keep.sum()), np.int8), (rp[keep], cp[keep])),
                                shape=(M, M)).tocsr()
    adj_pit_dir.sum_duplicates(); adj_pit_dir.data[:] = 1
    rc = np.asarray(rows_ct); cc2 = np.asarray(cols_ct); keep2 = rc != cc2
    adj_pit_contemp = sp.coo_matrix((np.ones(int(keep2.sum()), np.int8), (rc[keep2], cc2[keep2])),
                                    shape=(M, M)).tocsr()
    adj_pit_contemp.sum_duplicates(); adj_pit_contemp.data[:] = 1

    return TemporalGraph(
        node_time=node_time, asset_id=asset_id, adj=adj, adj_pit=adj_pit,
        adj_pit_dir=adj_pit_dir, adj_pit_contemp=adj_pit_contemp,
        features=np.nan_to_num(features).astype(np.float64),
        labels=np.nan_to_num(labels).astype(np.float64),
        meta=dict(market=market, n_assets=N, n_times=T, delta=delta,
                  edge_source=edge_source, k_neighbors=k_neighbors, days=days,
                  valid=valid, seed=seed),
    )
