import numpy as np
from math import log
from scipy.stats import norm


# -------- RNG --------
def make_rng(seed=None):
    """Create a reproducible NumPy Generator (PCG64)."""
    return np.random.default_rng(seed)


# -------- Core Sampler --------
def sample(dist, size, rng=None, **params):
    """
    Generic sampler for common distributions.

    dist in {
        'normal','lognormal','pareto','gamma','poisson','binomial',
        'negbin','exponential','weibull','mvnormal'
    }

    Note:
    - Discrete dists ('poisson','binomial','negbin') return integer dtype.
    - 'lognormal' supports `from_quantiles=True` with (x1,p1),(x2,p2).
    - 'negbin' expects `mean` (mu) and `k` (dispersion > 0) and uses a
      Gamma–Poisson mixture so k may be non-integer.
    """
    rng = rng or make_rng()
    n = int(size)
    if n <= 0:
        raise ValueError("size must be a positive integer")
    d = str(dist).lower()

    if d == "normal":
        loc = float(params["loc"])
        scale = float(params["scale"])
        if scale <= 0:
            raise ValueError("normal: scale must be > 0")
        return rng.normal(loc=loc, scale=scale, size=n)

    if d == "lognormal":
        if params.get("from_quantiles"):
            mu, sigma = _lognormal_mu_sigma_from_quantiles(
                x1=params["x1"], p1=params["p1"], x2=params["x2"], p2=params["p2"]
            )
        else:
            mu = float(params["meanlog"])
            sigma = float(params["sdlog"])
            if sigma <= 0:
                raise ValueError("lognormal: sdlog must be > 0")
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    if d == "pareto":
        a = float(params["alpha"])
        if a <= 0:
            raise ValueError("pareto: alpha must be > 0")
        xm = float(params.get("xm", 1.0))
        # NumPy's pareto is Lomax-1; this yields support [xm, inf)
        return xm * (1.0 + rng.pareto(a, size=n))

    if d == "gamma":
        shape = float(params["shape"])
        scale = float(params["scale"])
        if shape <= 0 or scale <= 0:
            raise ValueError("gamma: shape, scale must be > 0")
        return rng.gamma(shape=shape, scale=scale, size=n)

    if d == "poisson":
        lam = float(params["lam"])
        if lam < 0:
            raise ValueError("poisson: lam must be >= 0")
        # ensure integer dtype (NumPy already returns ints, make explicit)
        return rng.poisson(lam=lam, size=n).astype(np.int64)

    if d == "binomial":
        trials = int(params["n"])
        p = float(params["p"])
        if trials < 0 or not (0.0 <= p <= 1.0):
            raise ValueError("binomial: n >= 0 and 0 <= p <= 1 required")
        return rng.binomial(n=trials, p=p, size=n).astype(np.int64)

    if d == "negbin":
        # Gamma–Poisson mixture parameterization (robust for non-integer k)
        mu = float(params["mean"])
        k = float(params["k"])
        if mu < 0 or k <= 0:
            raise ValueError("negbin: mean >= 0 and k > 0 required")
        lam = rng.gamma(shape=k, scale=mu / k, size=n)
        return rng.poisson(lam).astype(np.int64)

    if d == "exponential":
        lam = float(params["lam"])
        if lam <= 0:
            raise ValueError("exponential: lam must be > 0")
        return rng.exponential(scale=1.0 / lam, size=n)

    if d == "weibull":
        k = float(params["k"])
        lam = float(params["lam"])
        if k <= 0 or lam <= 0:
            raise ValueError("weibull: k, lam must be > 0")
        # NumPy uses Weibull with scale=1; multiply by scale (lam)
        return lam * rng.weibull(a=k, size=n)

    if d == "mvnormal":
        mean = np.asarray(params["mean"], dtype=float)
        cov = np.asarray(params["cov"], dtype=float)
        if mean.ndim != 1 or cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mean.size:
            raise ValueError("mvnormal: mean (d,), cov (d,d) required")
        return rng.multivariate_normal(mean=mean, cov=cov, size=n)

    raise ValueError(f"Unsupported distribution: {dist!r}")


# -------- Helpers --------
def categorical(probs, size, rng=None, labels=None):
    """Draw from a categorical distribution (optionally return labels)."""
    rng = rng or make_rng()
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("categorical: probs must be a 1D non-empty array")
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("categorical: probs must sum to a positive finite value")
    p = p / s
    idx = rng.choice(len(p), size=int(size), p=p)
    if labels is None:
        return idx
    labels = np.asarray(labels, dtype=object)
    if labels.size != p.size:
        raise ValueError("categorical: labels length must match probs")
    return labels[idx]


def dirichlet(alpha, size=1, rng=None):
    """Draw Dirichlet vectors; returns shape (size, K)."""
    rng = rng or make_rng()
    a = np.asarray(alpha, dtype=float)
    if (a <= 0).any():
        raise ValueError("dirichlet: all alpha > 0 required")
    return rng.dirichlet(a, size=int(size))


def mixture(components, weights, size, rng=None):
    """
    Finite mixture sampler.

    components: list of (dist, params) tuples
    weights: list/array of floats (sum to 1)
    Returns float array of length `size`.
    """
    rng = rng or make_rng()
    w = np.asarray(weights, float)
    if w.ndim != 1 or w.size != len(components):
        raise ValueError("mixture: weights length must match components")
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("mixture: weights must sum to a positive finite value")
    w = w / s

    comp_idx = rng.choice(len(components), size=int(size), p=w)
    out = np.empty(int(size), dtype=float)
    for i, (dname, dparams) in enumerate(components):
        m = np.sum(comp_idx == i)
        if m:
            out[comp_idx == i] = sample(dname, m, rng=rng, **dparams)
    # result may mix ints/floats; keep float for generality
    return out


def survival_times(dist, size, rng=None, censor_at=None, **params):
    """
    Draw time-to-event data with optional right-censoring.
    dist in {'exponential','weibull','lognormal'}
    Returns (times, events) where events is 1 for observed, 0 for censored.
    """
    rng = rng or make_rng()
    d = str(dist).lower()
    if d == "lognormal":
        t = sample("lognormal", size, rng=rng, **params)
    elif d == "weibull":
        t = sample("weibull", size, rng=rng, **params)
    elif d == "exponential":
        t = sample("exponential", size, rng=rng, **params)
    else:
        raise ValueError("survival_times supports 'lognormal','weibull','exponential'")

    events = np.ones(int(size), dtype=int)
    if censor_at is not None:
        c = float(censor_at)
        cens = t > c
        t = t.copy()
        t[cens] = c
        events[cens] = 0
    return t, events


# -------- Internal --------
def _lognormal_mu_sigma_from_quantiles(x1, p1, x2, p2):
    """Solve (mu, sigma) of ln(X) ~ N(mu,sigma^2) from two quantiles."""
    x1 = float(x1); x2 = float(x2)
    p1 = float(p1); p2 = float(p2)
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("lognormal quantiles: p1,p2 must be in (0,1)")
    if not (0.0 < x1 < x2):
        raise ValueError("lognormal quantiles: require 0 < x1 < x2")

    z1 = norm.ppf(p1); z2 = norm.ppf(p2)
    dz = z2 - z1
    # New: reject nearly-equal percentiles (ill-conditioned)
    if not np.isfinite(z1) or not np.isfinite(z2) or abs(dz) < 1e-3:
        raise ValueError("lognormal quantiles: p1 and p2 too close or invalid")

    sigma = (log(x2) - log(x1)) / dz
    # New: sanity bound on sigma to avoid numerical blowups in RNG
    if not np.isfinite(sigma) or sigma <= 0 or sigma > 8.0:
        raise ValueError("lognormal quantiles: solved sigma invalid or too large")
    mu = log(x1) - sigma * z1
    return float(mu), float(sigma)
