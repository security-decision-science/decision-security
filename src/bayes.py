import numpy as np

def beta_update(a, b, successes, failures):
    """Conjugate update for Bernoulli likelihood with Beta(a,b) prior."""
    return float(a + successes), float(b + failures)

def normal_update_known_variance(mu0, sigma0, xbar, n, sigma):
    """
    Conjugate Normal-Normal update when observation variance sigma^2 is known.
    Returns (posterior_mean, posterior_std).
    """
    tau0 = 1.0 / (sigma0 ** 2)
    tau = 1.0 / (sigma ** 2)
    post_var = 1.0 / (tau0 + n * tau)
    post_mean = post_var * (tau0 * mu0 + n * tau * xbar)
    return float(post_mean), float(np.sqrt(post_var))

def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))

def inv_logit(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))

def brier_score(probs, outcomes):
    """Brier score for binary outcomes (lower is better)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - y) ** 2))

def calibration_curve(preds, outcomes, bins=10):
    """
    Reliability curve: average predicted prob vs empirical rate in bins.
    Returns (bin_centers, empirical_rates).
    """
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    qs = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(preds, qs) - 1, 0, bins - 1)
    emp = np.zeros(bins)
    cen = np.zeros(bins)
    for b in range(bins):
        mask = idx == b
        if mask.any():
            emp[b] = y[mask].mean()
            cen[b] = preds[mask].mean()
        else:
            emp[b] = np.nan
            cen[b] = (qs[b] + qs[b + 1]) / 2
    return cen, emp