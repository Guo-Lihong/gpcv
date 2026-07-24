"""Models over temporal relational graphs.

* ``diffuse`` / ``khop_predict`` : a deterministic L-step mean-aggregation
  predictor (a label-/feature-propagation GNN). No training, so it isolates the
  effect of graph DEPTH and adjacency on measured skill with zero optimisation
  noise — ideal for the leakage depth curve.
* ``GCN`` + ``train_gcn`` : a small trained graph-convolutional regressor, to
  confirm the same leakage under a learned model and to drive the CV-protocol
  comparison.

All ops use sparse adjacencies and run comfortably on CPU.
"""
import numpy as np
import scipy.sparse as sp

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Deterministic propagation predictor                                         #
# --------------------------------------------------------------------------- #
def row_normalize(adj):
    A = adj.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (sp.diags(dinv) @ A).tocsr()


def diffuse(adj, x, L):
    """L steps of row-normalised neighbour averaging (no self-loop at step 1)."""
    A = row_normalize(adj)
    h = np.asarray(x, dtype=np.float64).copy()
    for _ in range(int(L)):
        h = A @ h
    return h


def khop_predict(adj, features, L, feature_col=1):
    """Predict each node as the L-step neighbour-average of ``features[:, col]``."""
    x = np.asarray(features)[:, feature_col]
    return diffuse(adj, x, L)


# --------------------------------------------------------------------------- #
# Trained GCN                                                                 #
# --------------------------------------------------------------------------- #
def sym_normalize(adj, add_self=True):
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2 as a scipy CSR."""
    A = adj.tocsr().astype(np.float64)
    if add_self:
        A = A + sp.eye(A.shape[0], format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    D = sp.diags(dinv)
    return (D @ A @ D).tocsr()


def _to_torch_sparse(A):
    A = A.tocoo()
    idx = np.vstack([A.row, A.col])
    return torch.sparse_coo_tensor(
        torch.as_tensor(idx, dtype=torch.long),
        torch.as_tensor(A.data, dtype=torch.float32),
        size=A.shape,
    ).coalesce()


if _HAS_TORCH:

    class GNN(nn.Module):
        """Depth-L message-passing network. arch in {'gcn','sage','gin'}:
          gcn : h = W (P h),          P = normalised adjacency with self-loops
          sage: h = W_s h + W_n (P h), P = neighbour mean (no self-loop)  [GraphSAGE-mean]
          gin : h = W_s ((1+eps) h) + W_n (P h)                            [GIN-style]
        """
        def __init__(self, in_dim, hidden=32, depth=2, arch="gcn", dropout=0.0):
            super().__init__()
            self.depth = depth; self.arch = arch
            dims = [in_dim] + [hidden] * (depth - 1) + [1]
            if arch == "gcn":
                self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
            else:
                self.lin_self = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
                self.lin_nb = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
                if arch == "gin":
                    self.eps = nn.Parameter(torch.zeros(depth))
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, P, X):
            h = X
            for i in range(self.depth):
                agg = torch.sparse.mm(P, h)
                if self.arch == "gcn":
                    h = self.lins[i](agg)
                elif self.arch == "sage":
                    h = self.lin_self[i](h) + self.lin_nb[i](agg)
                else:  # gin
                    h = self.lin_self[i]((1.0 + self.eps[i]) * h) + self.lin_nb[i](agg)
                if i < self.depth - 1:
                    h = self.dropout(self.act(h))
            return h.squeeze(-1)

    def _propagation_matrix(adj, arch, directed, add_self, device):
        """Build the torch sparse propagation operator P for a given adjacency.

        Exactly the operator construction used by ``train_gcn`` (factored out so the
        fit-time and inference-time operators can differ).
        """
        if arch == "gcn":
            if directed:
                A = adj.tocsr().astype(np.float64)
                if add_self:
                    A = A + sp.eye(A.shape[0], format="csr")
                return _to_torch_sparse(row_normalize(A)).to(device)
            return _to_torch_sparse(sym_normalize(adj, add_self=add_self)).to(device)
        # sage / gin: neighbour mean (no self-loop); self handled inside the model
        return _to_torch_sparse(row_normalize(adj.tocsr().astype(np.float64))).to(device)

    def train_gcn(adj, features, labels, train_mask, depth=2, hidden=32,
                  epochs=200, lr=1e-2, weight_decay=5e-4, seed=0, add_self=True,
                  directed=False, standardize=True, device=None, task="reg",
                  pos_weight=None, arch="gcn", verbose=False, adj_eval=None):
        """Train a ``depth``-layer GNN by MSE (or BCE if task='clf') on ``train_mask``
        nodes; return predictions for ALL nodes (numpy).

        arch in {'gcn','sage','gin'}. directed=True uses row-normalised propagation
        (for the directed point-in-time operator); directed=False uses symmetric
        normalisation for gcn.

        ``adj_eval`` (optional): if given, the final forward pass that produces the
        returned predictions propagates over ``adj_eval`` instead of ``adj``, using
        the identical normalisation. Fitting is unchanged. When ``adj_eval is None``
        the behaviour is byte-identical to propagating over ``adj`` throughout.
        """
        torch.manual_seed(seed)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        P = _propagation_matrix(adj, arch, directed, add_self, device)

        X = np.asarray(features, dtype=np.float64)
        if standardize:
            mu = X[train_mask].mean(0); sd = X[train_mask].std(0) + 1e-8
            X = (X - mu) / sd
        X = torch.as_tensor(X, dtype=torch.float32).to(device)
        y = torch.as_tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float32).to(device)
        tm = torch.as_tensor(np.asarray(train_mask, dtype=bool)).to(device)
        ym = y[tm]
        y_mu, y_sd = ym.mean(), ym.std() + 1e-8

        model = GNN(X.shape[1], hidden=hidden, depth=depth, arch=arch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if task == "clf":
            pw = None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            lossf = nn.MSELoss()
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            out = model(P, X)
            target = ym if task == "clf" else (ym - y_mu) / y_sd
            loss = lossf(out[tm], target)
            loss.backward()
            opt.step()
            if verbose and ep % 50 == 0:
                print(f"  epoch {ep} loss {loss.item():.4f}")
        model.eval()
        P_eval = P if adj_eval is None else _propagation_matrix(
            adj_eval, arch, directed, add_self, device)
        with torch.no_grad():
            pred = model(P_eval, X).cpu().numpy()
        return pred
