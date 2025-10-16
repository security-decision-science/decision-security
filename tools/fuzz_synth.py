#!/usr/bin/env python3
"""
Minimal fuzzer for decision_security.synth and friends.

- Randomizes valid-ish inputs.
- Asserts basic invariants (no crash, shapes, finiteness, constraints).
- Allows ValueError / NotImplementedError for unsupported param combos.
- Time-boxed and iteration-limited.
"""
from __future__ import annotations
import argparse, math, sys, time, random
import numpy as np

def _have(modname: str, attr: str) -> bool:
    try:
        mod = __import__(modname, fromlist=[attr])
        return hasattr(mod, attr)
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=400, help="number of fuzz cases")
    ap.add_argument("--timeout-sec", type=float, default=60, help="overall time budget")
    ap.add_argument("--seed", type=int, default=123, help="PRNG seed")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    failures = 0
    checked = 0

    try:
        from decision_security import synth as S
    except Exception as e:
        print("FATAL: cannot import decision_security.synth:", repr(e), file=sys.stderr)
        return 2

    # what features exist?
    HAS = {
        "sample": hasattr(S, "sample"),
        "make_rng": hasattr(S, "make_rng"),
        "categorical": hasattr(S, "categorical"),
        "mixture": hasattr(S, "mixture"),
        "survival_times": hasattr(S, "survival_times"),
        "dirichlet": hasattr(S, "dirichlet"),
    }
    if not HAS["make_rng"]:
        def make_rng(seed=None): return np.random.default_rng(seed)
    else:
        make_rng = S.make_rng

    while checked < args.iterations and (time.time() - t0) < args.timeout_sec:
        checked += 1
        rng = make_rng(random.randrange(0, 2**32-1))

        try:
            # Pick an op
            op = random.choice(
                [k for k,v in HAS.items() if v and k in
                 ("sample","categorical","mixture","survival_times","dirichlet")]
            )

            if op == "sample":
                # Choose a supported dist; prefer ones likely implemented
                dist = random.choice(["poisson","lognormal","negbin","normal"])
                n = random.randint(1, 5000)
                if dist == "poisson":
                    lam = 10.0 ** random.uniform(-2, 1.2)  # ~0.01 .. ~15
                    x = S.sample("poisson", n, rng=rng, lam=lam)
                    assert isinstance(x, np.ndarray) and x.shape == (n,)
                    assert np.isfinite(x).all() and (x >= 0).all()
                    assert np.issubdtype(x.dtype, np.integer)
                elif dist == "lognormal":
                    # Try quantile-based parameterization
                    x1, x2 = sorted(10.0 ** np.random.uniform(-1, 2, size=2))
                    p1, p2 = sorted(np.random.uniform(0.1, 0.99, size=2))
                    x = S.sample("lognormal", n, rng=rng, from_quantiles=True, x1=x1, p1=p1, x2=x2, p2=p2)
                    assert x.shape == (n,) and np.isfinite(x).all() and (x > 0).all()
                elif dist == "negbin":
                    mu = 10.0 ** random.uniform(-2, 1.3)  # ~0.01 .. 20
                    k  = 10.0 ** random.uniform(-2, 1.0)  # dispersion > 0
                    x = S.sample("negbin", n, rng=rng, mean=mu, k=k)
                    assert x.shape == (n,) and np.isfinite(x).all()
                    assert (x >= 0).all() and np.mean(x) >= 0
                elif dist == "normal":
                    loc = random.uniform(-5, 5); scale = 10.0 ** random.uniform(-2, 1)
                    x = S.sample("normal", n, rng=rng, loc=loc, scale=scale)
                    assert x.shape == (n,) and np.isfinite(x).all()

            elif op == "categorical":
                k = random.randint(2, 6)
                w = np.abs(np.random.default_rng().normal(size=k))
                w = w / w.sum()
                size = random.randint(1, 4000)
                labels = [f"L{i}" for i in range(k)]
                lbls = S.categorical(w, size=size, rng=rng, labels=labels)
                assert len(lbls) == size
                assert set(lbls).issubset(set(labels))

            elif op == "mixture":
                # two-component normal mixture
                comps = [("normal", {"loc": 0.0, "scale": 1.0}),
                         ("normal", {"loc": random.uniform(2, 12), "scale": 1.0})]
                weights = [0.5, 0.5]
                size = random.randint(10, 10000)
                x = S.mixture(components=comps, weights=weights, size=size, rng=rng)
                assert isinstance(x, np.ndarray) and x.shape == (size,)
                assert np.isfinite(x).all()

            elif op == "survival_times":
                # Exponential baseline with optional censoring
                lam = 10.0 ** random.uniform(-3, -0.3)  # hazard ~0.001..0.5
                size = random.randint(10, 5000)
                censor_at = random.choice([None, random.uniform(1.0, 20.0)])
                t, e = S.survival_times("exponential", size, rng=rng, lam=lam, censor_at=censor_at)
                assert len(t) == size and len(e) == size
                assert np.isfinite(t).all() and (t >= 0).all()
                assert set(np.unique(e)).issubset({0,1})
                if censor_at is not None:
                    assert (t <= censor_at + 1e-9).all()

            elif op == "dirichlet":
                alpha = 10.0 ** np.random.uniform(-2, 1, size=random.randint(2, 6))
                x = S.dirichlet(alpha, rng=rng)
                assert np.isfinite(x).all() and np.all(x >= 0)
                s = float(np.sum(x))
                assert math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6)

        except (ValueError, NotImplementedError, AssertionError) as e:
            # Count failures only for violated invariants; unsupported combos are ok.
            if isinstance(e, AssertionError):
                failures += 1
                print(f"[FAIL] {op}: {e}", file=sys.stderr)
        except Exception as e:
            failures += 1
            print(f"[CRASH] {type(e).__name__}: {e}", file=sys.stderr)

    print(f"Fuzzed {checked} cases; failures={failures}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
