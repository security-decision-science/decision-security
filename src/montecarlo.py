import numpy as np
from numpy.random import default_rng

def make_lognormal_severity(meanlog, sdlog):
    """
    Return a severity sampler function for lognormal losses.

    Returns
    -------
    sampler(n: int, rng: numpy.random.Generator|None) -> np.ndarray
    """
    def _sampler(n, rng=None):
        r = rng or default_rng()
        return r.lognormal(mean=meanlog, sigma=sdlog, size=n)
    return _sampler

def simulate_aggregate_losses(n_periods, lam, severity_sampler, rng=None):
    """
    Compound Poisson aggregate loss simulator.
    For each period: N ~ Poisson(lam); if N>0, sum severity_sampler(N).

    Parameters
    ----------
    n_periods : int
    lam : float
    severity_sampler : callable (n, rng=None) -> np.ndarray
    rng : numpy.random.Generator or None
    """
    r = rng or default_rng()
    totals = np.zeros(n_periods, dtype=float)
    for i in range(n_periods):
        n = r.poisson(lam)
        if n > 0:
            totals[i] = float(np.sum(severity_sampler(n, r)))
    return totals

def risk_bands(losses, qs=(0.5, 0.9, 0.95)):
    """Return dict of quantiles like {'p50': ..., 'p90': ..., 'p95': ...}."""
    arr = np.asarray(list(losses), dtype=float)
    vals = np.quantile(arr, qs)
    return {f"p{int(q*100)}": float(v) for q, v in zip(qs, vals)}

def var_es(losses, alpha=0.95):
    """
    Value-at-Risk and Expected Shortfall at level alpha.
    Returns (VaR, ES).
    """
    arr = np.sort(np.asarray(list(losses), dtype=float))
    if arr.size == 0:
        return (0.0, 0.0)
    idx = int(np.ceil(alpha * len(arr))) - 1
    var = float(arr[max(0, idx)])
    es = float(arr[max(0, idx):].mean()) if idx < len(arr) else var
    return var, es