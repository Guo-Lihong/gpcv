"""Evaluation metrics: cross-sectional IC/RankIC, Deflated Sharpe, PBO, bootstrap.

All backtest-validity metrics follow Bailey & Lopez de Prado (Deflated Sharpe
Ratio; Probability of Backtest Overfitting via CSCV).
"""
from itertools import combinations
from math import sqrt

import numpy as np
from scipy.stats import norm, rankdata


# --------------------------------------------------------------------------- #
# Cross-sectional information coefficient                                      #
# --------------------------------------------------------------------------- #
def _corr(a, b, rank):
    if a.size < 3:
        return np.nan
    if rank:
        a = rankdata(a)
        b = rankdata(b)
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a * a).sum())
    db = np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return np.nan
    return float((a * b).sum() / (da * db))


def cross_sectional_ic(pred, y, node_time, mask=None, rank=True):
    """Per-time cross-sectional correlation between prediction and label.

    Returns the array of per-date ICs (RankIC if ``rank``). Mean is the IC;
    mean/std is the ICIR.
    """
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    node_time = np.asarray(node_time)
    sel = np.ones_like(node_time, dtype=bool) if mask is None else np.asarray(mask, bool)
    out = []
    for t in np.unique(node_time[sel]):
        m = sel & (node_time == t)
        c = _corr(pred[m], y[m], rank)
        if not np.isnan(c):
            out.append(c)
    return np.array(out)


def ic_summary(pred, y, node_time, mask=None):
    rank = cross_sectional_ic(pred, y, node_time, mask, rank=True)
    lin = cross_sectional_ic(pred, y, node_time, mask, rank=False)
    def _s(x):
        return dict(mean=float(np.mean(x)) if x.size else np.nan,
                    std=float(np.std(x)) if x.size else np.nan,
                    icir=float(np.mean(x) / np.std(x)) if x.size and np.std(x) > 0 else np.nan,
                    n=int(x.size))
    return dict(rankic=_s(rank), ic=_s(lin))


def binary_clf_metrics(logits, y, mask):
    """AUC / average-precision / F1 (at logit>0) for the positive class on ``mask``."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    p = np.asarray(logits, float)[mask]
    t = np.asarray(y, float)[mask].astype(int)
    if t.sum() == 0 or t.sum() == t.size:
        return dict(auc=np.nan, ap=np.nan, f1=np.nan, n=int(t.size), pos=int(t.sum()))
    return dict(auc=float(roc_auc_score(t, p)),
                ap=float(average_precision_score(t, p)),
                f1=float(f1_score(t, (p > 0).astype(int))),
                n=int(t.size), pos=int(t.sum()))


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of ``values``."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Long-short portfolio                                                         #
# --------------------------------------------------------------------------- #
def decile_long_short(pred, y, node_time, mask=None, q=0.2):
    """Per-date return of a top-vs-bottom quantile long-short book on the label.

    ``y`` here is the realized forward return. Returns (per_date_returns, sharpe).
    """
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    node_time = np.asarray(node_time)
    sel = np.ones_like(node_time, dtype=bool) if mask is None else np.asarray(mask, bool)
    rets = []
    for t in np.unique(node_time[sel]):
        m = sel & (node_time == t)
        if m.sum() < 5:
            continue
        p = pred[m]; r = y[m]
        k = max(1, int(np.ceil(q * p.size)))
        order = np.argsort(p)
        short = r[order[:k]].mean()
        long = r[order[-k:]].mean()
        rets.append(long - short)
    rets = np.array(rets)
    sharpe = float(rets.mean() / rets.std() * sqrt(252)) if rets.size and rets.std() > 0 else np.nan
    return rets, sharpe


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio                                                        #
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(sr, n_obs, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """PSR: P(true SR > benchmark) given observed SR and higher moments.

    ``sr`` and ``sr_benchmark`` are per-observation (non-annualized) Sharpe.
    """
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * sqrt(max(1, n_obs - 1)) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(sr, n_obs, n_trials, sr_variance, skew=0.0, kurt=3.0):
    """DSR: PSR deflated by the expected maximum SR under ``n_trials`` trials.

    ``sr_variance`` = variance of the Sharpe estimates across trials.
    """
    n_trials = max(2, int(n_trials))
    e = np.euler_gamma
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_star = sqrt(max(1e-12, sr_variance)) * ((1 - e) * z1 + e * z2)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_benchmark=sr_star)


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)                                   #
# --------------------------------------------------------------------------- #
def probability_backtest_overfitting(perf, n_splits=10):
    """PBO via Combinatorially-Symmetric Cross-Validation (Bailey et al. 2017).

    Parameters
    ----------
    perf : array (T_obs, N_trials) of per-period performance for each trial/config.
    n_splits : even number S of contiguous submatrices.

    Returns the probability that the in-sample-best config is below-median OOS.
    """
    perf = np.asarray(perf, float)
    T, N = perf.shape
    if N < 2:
        return np.nan
    S = n_splits - (n_splits % 2)
    S = max(2, min(S, T))
    bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [perf[bounds[i]:bounds[i + 1]] for i in range(S)]
    logits = []
    for is_idx in combinations(range(S), S // 2):
        oos_idx = [i for i in range(S) if i not in is_idx]
        IS = np.vstack([blocks[i] for i in is_idx])
        OOS = np.vstack([blocks[i] for i in oos_idx])
        is_perf = IS.mean(axis=0)
        oos_perf = OOS.mean(axis=0)
        n_star = int(np.argmax(is_perf))
        # OOS rank (fractional) of the IS-best config
        r = rankdata(oos_perf)[n_star] / (N + 1)
        r = min(max(r, 1e-6), 1 - 1e-6)
        logits.append(np.log(r / (1 - r)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0))
