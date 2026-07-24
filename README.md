# gpcv — Graph-Purged Cross-Validation

Leakage-safe cross-validation for graph and relational machine learning on time-stamped data.

Purged and embargoed cross-validation removes label-overlap leakage for tabular models. It does not
close a second channel that opens as soon as a model aggregates over a graph: an `L`-layer
message-passing network builds each node's representation from every node within `L` hops, and when
those neighbours carry timestamps on the other side of the train/test boundary, future information
reaches the model through edges rather than through any single date.

For a network of depth `L` on a graph whose edges bridge at most `δ` in time, the margin that must be
cleared around the boundary is

```
R(L) = h + L·δ + e          h = label horizon, e = embargo
```

Temporal purging removes only `h`. The `L·δ` term grows with depth, and is what this package measures
and corrects.

## Install

Python 3.8+. Install the dependencies and put `src/` on the path:

```bash
pip install numpy scipy scikit-learn      # core
pip install torch                         # only for the GNN models in gpcv.models
```

```python
import sys; sys.path.insert(0, "src")
import gpcv
```

## Diagnose

Audit an existing graph — no training, no labels, no model required.

```python
import gpcv

# How far in time does a single edge reach?  -> dict with delta_max, delta_mean, quantiles
gpcv.edge_temporal_reach(adj, node_time)
# {'n_edges': 59942, 'delta_max': 5, 'delta_mean': 2.70, ...}

# What fraction of nodes can see the future within L hops?  -> dict with frac_future
gpcv.future_reach_fraction(adj, node_time, L=3)

# How many training nodes stay contaminated as depth grows?  -> one row per depth
gpcv.contamination_curve(adj, node_time, test_mask, L_max=5, h=1, embargo=2)
# [{'L': 1, 'frac_contaminated': 0.040, 'margin_days': 5},
#  {'L': 3, 'frac_contaminated': 0.218, 'margin_days': 15}, ...]
```

`margin_days` is `L·δ`, the reach a temporal purge does not cover.

## Correct

```python
from gpcv import make_folds

folds = make_folds("graph", node_time, adj, L=3,
                   label_horizon=1, embargo=2,
                   n_blocks=6, n_test_blocks=2)

for train_mask, test_mask in folds:
    ...
```

`make_folds` takes `"temporal"` (label-overlap purge + embargo) or `"graph"` (adds the `L`-hop
neighbourhood purge). The individual masks are available as `temporal_purge_mask`,
`graph_purge_mask` and `l_hop_reach`.

Two things matter in practice, and only one of them is a splitter.

**Use a directed, time-respecting operator.** Building edges point-in-time is not enough: an
undirected graph re-admits the future at any depth `L ≥ 2`, because a two-hop walk can land on a later
timestamp even when every edge was constructed strictly in the past. In our experiments this operator,
not the purge margin, does the work.

**Budget the margin for depth.** `R(L)` grows with `L`. Where a directed operator does not apply —
static or genuinely undirected graphs — purge the `L`-hop neighbourhood of the test set. Note that the
embargo already subsumes that purge until `L·δ > h + e`.

## Modules

```
gpcv/splitters.py     temporal purge, embargo, L-hop purge, CPCV blocks, make_folds
gpcv/diagnostics.py   edge_temporal_reach, future_reach_fraction, contamination_curve
gpcv/models.py        message-passing models (GCN / GraphSAGE / GIN); requires torch
gpcv/metrics.py       rank-IC, long-short backtest, PSR / DSR / PBO
gpcv/pathboot.py      stationary block bootstrap for path statistics (drawdown, Sortino)
gpcv/synthetic.py     controlled temporal-graph generator with known ground truth
gpcv/rsr.py           loader for the RSR equity dataset
gpcv/elliptic.py      loader for the Elliptic Bitcoin transaction graph
```

`gpcv.synthetic` builds graphs with known ground truth, so the diagnostics can be exercised without
any external data. The two loaders expect datasets that are not redistributed here: **RSR** (Feng et
al., 2019) is AGPL-3.0 and must be obtained from its authors; **Elliptic** has its own distribution
terms.

