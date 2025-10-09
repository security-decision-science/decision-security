import numpy as np
from numpy.random import default_rng
from math import log
from scipy.stats import norm

def make_rng(seed=None):
    """Create a reproducible NumPy Generator."""
    return np.random.default_rng(seed)

def sample(dist, size, rng=None, **params):
    """
    Generic sampler for common distributions.
    dist in {'normal','lognormal','pareto','gamma','poisson','binomial',
             'negbin','exponential','weibull','mvnormal'}
    """
    rng = rng or make_rng()
    d = dist.lower()

    if d == "normal":
        return rng.normal(loc=params["loc"], scale=params["scale"], size=size)

    if d == "lognormal":
        if params.get("from_quantiles"):
            mu, sigma = _lognormal_mu_sigma_from_quantiles(
                x1=params["x1"], p1=params["p1"], x2=params["x2"], p2=params["p2"]
            )
        else:
            mu, sigma = params["meanlog"], params["sdlog"]
        return rng.lognormal(mean=mu, sigma=sigma, size=size)

    if d == "pareto":
        a = params["alpha"]; xm = params.get("xm", 1.0)
        return xm * (1.0 + rng.pareto(a, size=size))

    if d == "gamma":
        return rng.gamma(shape=params["shape"], scale=params["scale"], size=size)

    if d == "poisson":
        return rng.poisson(lam=params["lam"], size=size)

    if d == "binomial":
        return rng.binomial(n=int(params["n"]), p=float(params["p"]), size=size)

    if d == "negbin":
        mu, k = float(params["mean"]), float(params["k"])
        p = k / (k + mu); n = k
        return rng.negative_binomial(n=n, p=p, size=size)

    if d == "exponential":
        lam = float(params["lam"])
        return rng.exponential(scale=1.0 / lam, size=size)

    if d == "weibull":
        k = float(params["k"]); lam = float(params["lam"])
        return lam * rng.weibull(a=k, size=size)

    if d == "mvnormal":
        mean = np.asarray(params["mean"], dtype=float)
        cov = np.asarray(params["cov"], dtype=float)
        return rng.multivariate_normal(mean=mean, cov=cov, size=size)

    raise ValueError("Unsupported distribution: %s" % dist)

def categorical(probs, size, rng=None, labels=None):
    """Draw from a categorical distribution (optionally return labels)."""
    rng = rng or make_rng()
    p = np.asarray(probs, dtype=float); p = p / p.sum()
    idx = rng.choice(len(p), size=size, p=p)
    if labels is None:
        return idx
    labels = np.asarray(labels, dtype=object)
    return labels[idx]

def dirichlet(alpha, size=1, rng=None):
    """Draw Dirichlet vectors (size x K)."""
    rng = rng or make_rng()
    return rng.dirichlet(alpha, size=size)

def mixture(components, weights, size, rng=None):
    """
    Finite mixture sampler.

    components: list of (dist, params) tuples
    weights: list of floats (sum to 1)
    """
    rng = rng or make_rng()
    w = np.asarray(weights, float); w = w / w.sum()
    comp_idx = rng.choice(len(components), size=size, p=w)
    out = np.empty(size, dtype=float)
    for i, (dist, params) in enumerate(components):
        m = np.sum(comp_idx == i)
        if m:
            out[comp_idx == i] = sample(dist, m, rng=rng, **params)
    return out

def survival_times(dist, size, rng=None, censor_at=None, **params):
    """
    Draw time-to-event data with optional right-censoring.
    dist in {'exponential','weibull','lognormal'}
    Returns (times, events) where events is 1 for observed, 0 for censored.
    """
    rng = rng or make_rng()
    if dist == "lognormal":
        t = sample("lognormal", size, rng=rng, **params)
    elif dist == "weibull":
        t = sample("weibull", size, rng=rng, **params)
    elif dist == "exponential":
        t = sample("exponential", size, rng=rng, **params)
    else:
        raise ValueError("survival_times supports 'lognormal','weibull','exponential'")

    events = np.ones(size, dtype=int)
    if censor_at is not None:
        cens = t > censor_at
        t[cens] = censor_at
        events[cens] = 0
    return t, events

def _lognormal_mu_sigma_from_quantiles(x1, p1, x2, p2):
    """Solve (mu, sigma) of ln(X) ~ N(mu,sigma^2) from two quantiles."""
    z1 = norm.ppf(p1); z2 = norm.ppf(p2)
    if not (0 < x1 < x2):
        raise ValueError("Require 0 < x1 < x2 for lognormal quantiles.")
    sigma = (log(x2) - log(x1)) / (z2 - z1)
    mu = log(x1) - sigma * z1
    return float(mu), float(sigma)