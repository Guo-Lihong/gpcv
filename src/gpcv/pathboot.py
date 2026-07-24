"""Path statistics and a stationary block bootstrap for them.

Why this module exists
----------------------
``gpcv.metrics.bootstrap_ci`` is an i.i.d. bootstrap of the MEAN. It is valid only for a
statistic that is an average of (approximately) independent observations. It is NOT valid
for PATH statistics -- maximum drawdown, Calmar, ulcer index, time under water -- which
depend on the ORDER of the series, nor for ratios of such statistics, nor for Sortino /
Sharpe when the daily series is serially dependent. Resampling days independently destroys
exactly the dependence that produces a drawdown.

This module therefore provides

  * vectorised path statistics that operate row-wise on a (m, T) matrix of daily returns;
  * the stationary block bootstrap of Politis & Romano (1994): blocks with geometric length
    (mean block ``B``, i.e. p = 1/B) wrapped circularly, which preserves short-range serial
    dependence and yields a stationary resampled series.

Caveat stated for the record: a bootstrap for max drawdown is an inference about the
drawdown DISTRIBUTION of a stationary process with the estimated dependence structure. It
is not a confidence interval for "the drawdown that happened", which is a realised path
quantity observed without error. All intervals below should be read that way.
"""
import numpy as np

ANN = np.sqrt(252.0)
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Vectorised path statistics: R is (m, T) or (T,)                              #
# --------------------------------------------------------------------------- #
def _as2d(R):
    R = np.asarray(R, float)
    return R[None, :] if R.ndim == 1 else R


def equity_curve(R):
    """Compounded equity from daily simple returns, starting at 1.0."""
    return np.cumprod(1.0 + _as2d(R), axis=1)


def drawdown_curve(R):
    eq = equity_curve(R)
    return eq / np.maximum.accumulate(eq, axis=1) - 1.0


def max_drawdown(R):
    """Most negative point of the drawdown curve (a negative number)."""
    return drawdown_curve(R).min(axis=1)


def ann_return(R):
    """Arithmetic annualised mean (mu * 252), matching the rest of the codebase."""
    return _as2d(R).mean(axis=1) * TRADING_DAYS


def ann_vol(R):
    return _as2d(R).std(axis=1) * ANN


def sharpe(R):
    R = _as2d(R)
    sd = R.std(axis=1)
    return np.where(sd > 0, R.mean(axis=1) / np.where(sd > 0, sd, 1.0) * ANN, np.nan)


def downside_deviation(R):
    """MAR=0 downside deviation: sqrt(mean(min(r,0)^2)) over ALL days, annualised.

    This is the Sortino (1994) / Bacon definition. Using ``std`` of only the negative
    returns -- as an earlier version of the analysis did -- is a different and much smaller
    quantity: it both drops the positive days from the denominator's sample size and
    subtracts the mean of the negatives, so it inflates the resulting Sortino ratio.
    """
    R = _as2d(R)
    neg = np.minimum(R, 0.0)
    return np.sqrt((neg * neg).mean(axis=1)) * ANN


def sortino(R):
    dd = downside_deviation(R)
    return np.where(dd > 0, ann_return(R) / np.where(dd > 0, dd, 1.0), np.nan)


def calmar(R):
    mdd = np.abs(max_drawdown(R))
    return np.where(mdd > 0, ann_return(R) / np.where(mdd > 0, mdd, 1.0), np.nan)


def ulcer_index(R):
    """sqrt(mean(dd_t^2)) over the whole path, in return units (multiply by 100 for %)."""
    d = drawdown_curve(R)
    return np.sqrt((d * d).mean(axis=1))


def underwater_stats(r, tol=1e-12):
    """Time-under-water statistics for a SINGLE series (not vectorised).

    Returns max / mean underwater spell length in days, the number of spells, the fraction
    of days under water, and the time-to-recovery of the maximum drawdown: days from the
    drawdown trough until equity regains its prior peak (None if never recovered).
    """
    r = np.asarray(r, float)
    d = drawdown_curve(r)[0]
    under = d < -tol
    spells, start = [], None
    for i, u in enumerate(under):
        if u and start is None:
            start = i
        elif not u and start is not None:
            spells.append(i - start)
            start = None
    unrecovered = None
    if start is not None:                       # still under water at the end
        spells.append(len(under) - start)
        unrecovered = len(under) - start
    trough = int(np.argmin(d))
    rec = np.where(~under[trough:])[0]
    ttr = int(rec[0]) if rec.size else None     # days from trough back to the old peak
    return dict(
        max_underwater_days=int(max(spells)) if spells else 0,
        mean_underwater_days=float(np.mean(spells)) if spells else 0.0,
        n_underwater_spells=int(len(spells)),
        frac_days_underwater=float(under.mean()),
        time_to_recovery_days=ttr,
        max_dd_trough_day=trough,
        still_underwater_days_at_end=unrecovered,
    )


def var_cvar(r, q=5.0, demean=False):
    """Historical VaR/CVaR at the ``q``-th percentile of a single daily series.

    ``demean=True`` removes the series' own mean first, which separates the tail SHAPE
    from the drift. A high-drift series has a mechanically less negative raw VaR even when
    its shocks are identical.
    """
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if demean:
        r = r - r.mean()
    v = float(np.percentile(r, q))
    tail = r[r <= v]
    return v, float(tail.mean()) if tail.size else float("nan")


# --------------------------------------------------------------------------- #
# Stationary block bootstrap (Politis & Romano 1994)                           #
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(n, n_reps, mean_block=21, seed=0):
    """(n_reps, n) integer index matrix for the stationary bootstrap.

    Each resampled series starts at a uniformly random position; with probability
    p = 1/mean_block the next position restarts at a fresh uniform draw, otherwise it
    advances by one (mod n). Block lengths are therefore Geometric(p) with mean
    ``mean_block`` and the resampled series is strictly stationary.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / float(mean_block)
    idx = np.empty((n_reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_reps)
    restart = rng.random((n_reps, n)) < p
    jump = rng.integers(0, n, size=(n_reps, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], jump[:, t], cont)
    return idx


def bootstrap_paths(stat_fn, n_obs, n_reps=4000, mean_block=21, alpha=0.10, seed=0,
                    batch=250):
    """Percentile CI for arbitrary path statistics under the stationary bootstrap.

    ``stat_fn(idx)`` takes an (b, n_obs) index matrix and returns a dict {name: (b,) array}.
    The point estimate is ``stat_fn`` evaluated on the identity index. Returns
    {name: dict(point, lo, hi, se, mean)} with a (1-alpha) two-sided percentile interval.
    """
    ident = np.arange(n_obs)[None, :]
    point = {k: float(np.asarray(v).ravel()[0]) for k, v in stat_fn(ident).items()}
    idx = stationary_bootstrap_indices(n_obs, n_reps, mean_block, seed)
    acc = {k: [] for k in point}
    for i in range(0, n_reps, batch):
        out = stat_fn(idx[i:i + batch])
        for k in acc:
            acc[k].append(np.asarray(out[k], float).ravel())
    lo_q, hi_q = 100 * alpha / 2.0, 100 * (1 - alpha / 2.0)
    res = {}
    for k, chunks in acc.items():
        d = np.concatenate(chunks)
        d = d[np.isfinite(d)]
        if d.size == 0:
            res[k] = dict(point=point[k], lo=float("nan"), hi=float("nan"),
                          se=float("nan"), boot_mean=float("nan"), n_reps=0)
            continue
        res[k] = dict(point=point[k], lo=float(np.percentile(d, lo_q)),
                      hi=float(np.percentile(d, hi_q)), se=float(d.std()),
                      boot_mean=float(d.mean()), n_reps=int(d.size))
    return res


def bootstrap_mean_ci(x, n_reps=4000, mean_block=21, alpha=0.10, seed=0):
    """Stationary-bootstrap CI for the MEAN of a serially dependent series.

    Unlike an i.i.d. bootstrap of the mean this keeps the block structure, so it is the
    right tool for a per-date IC or gap series.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return dict(point=float("nan"), lo=float("nan"), hi=float("nan"),
                    se=float("nan"), boot_mean=float("nan"), n_reps=0)
    return bootstrap_paths(lambda I: {"mean": x[I].mean(axis=1)}, x.size,
                           n_reps, mean_block, alpha, seed)["mean"]
