import numpy as np
from decision_security.synth import make_rng, sample, mixture, categorical, survival_times, dirichlet

def test_reproducible():
    a = sample("poisson", 5, rng=make_rng(1), lam=2.0)
    b = sample("poisson", 5, rng=make_rng(1), lam=2.0)
    assert np.all(a == b)

def test_lognormal_quantiles():
    x = sample("lognormal", 50_000, rng=make_rng(0), from_quantiles=True, x1=10, p1=0.5, x2=100, p2=0.95)
    p50, p95 = np.quantile(x, [0.50, 0.95])
    assert 8 < p50 < 12 and 80 < p95 < 120

def test_negbin_moments():
    mu, k = 0.8, 3.0
    x = sample("negbin", 100_000, rng=make_rng(0), mean=mu, k=k)
    assert abs(x.mean() - mu) < 0.05
    assert x.var() > x.mean()  # over-dispersed

def test_categorical_labels():
    lbls = categorical([0.7, 0.3], size=10, rng=make_rng(0), labels=["A","B"])
    assert set(lbls).issubset({"A","B"})

def test_mixture_basic():
    x = mixture(
        components=[("normal", {"loc":0,"scale":1}), ("normal", {"loc":10,"scale":1})],
        weights=[0.5,0.5], size=10_000, rng=make_rng(0)
    )
    assert 3 < x.mean() < 7

def test_survival_censoring():
    t, e = survival_times("exponential", 1000, rng=make_rng(0), lam=0.05, censor_at=10.0)
    assert (t <= 10.0).all()
    assert set(e).issubset({0,1})