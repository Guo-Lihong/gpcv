"""Load the Elliptic Bitcoin dataset into a `TemporalGraph` (transductive channel).

Elliptic is a temporal transaction graph (203,769 nodes, 49 time steps). We use it to
demonstrate the depth-dependent TRANSDUCTIVE message-passing leakage: an undirected graph
lets training-time message passing reach test-period nodes, inflating illicit-class
detection; a directed point-in-time operator (a node aggregates only sources at time <= its
own) removes it. This extends "When Graph Structure Becomes a Liability" with the depth law
and the CPCV fix.

Data: Weber et al. (2019). Downloaded via PyG's mirror; not redistributed here.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .synthetic import TemporalGraph


def build_elliptic_temporal_graph(raw_dir):
    feats = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None).values
    txid = feats[:, 0].astype(np.int64)
    node_time = feats[:, 1].astype(np.int64)
    X = feats[:, 2:].astype(np.float64)
    M = len(txid)
    id2idx = {int(t): i for i, t in enumerate(txid)}

    cls = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_classes.csv"))
    lab = np.full(M, -1, dtype=np.int64)
    for tx, c in zip(cls["txId"].values, cls["class"].values):
        i = id2idx.get(int(tx))
        if i is None:
            continue
        cs = str(c)
        if cs == "1":       # illicit
            lab[i] = 1
        elif cs == "2":     # licit
            lab[i] = 0

    el = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_edgelist.csv"))
    src = el.iloc[:, 0].map(id2idx).to_numpy()
    dst = el.iloc[:, 1].map(id2idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64)

    # undirected (leaky / transductive) adjacency
    rr = np.concatenate([src, dst]); cc = np.concatenate([dst, src])
    adj = sp.coo_matrix((np.ones(rr.size, np.int8), (rr, cc)), shape=(M, M)).tocsr()
    adj.setdiag(0); adj.eliminate_zeros(); adj.sum_duplicates(); adj.data[:] = 1

    # directed point-in-time operator [target, source] with source_time <= target_time
    ta = node_time[src]; tb = node_time[dst]
    ab = ta <= tb   # a is source, b is target
    ba = tb <= ta   # b is source, a is target
    d_rows = np.concatenate([dst[ab], src[ba]])
    d_cols = np.concatenate([src[ab], dst[ba]])
    good = d_rows != d_cols
    adj_dir = sp.coo_matrix((np.ones(int(good.sum()), np.int8), (d_rows[good], d_cols[good])),
                            shape=(M, M)).tocsr()
    adj_dir.sum_duplicates(); adj_dir.data[:] = 1

    return TemporalGraph(
        node_time=node_time, asset_id=np.arange(M), adj=adj, adj_pit=adj,
        adj_pit_dir=adj_dir, features=np.nan_to_num(X),
        labels=lab.astype(np.float64),
        meta=dict(dataset="elliptic", n_nodes=M, valid=(lab >= 0),
                  n_times=int(node_time.max())),
    )
