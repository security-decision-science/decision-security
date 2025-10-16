#!/usr/bin/env python3
"""
Minimal fuzzer for decision_security.synth and friends.

- Randomizes valid-ish inputs.
- Asserts basic invariants (no crash, shapes, finiteness, constraints).
- Allows ValueError / NotImplementedError for unsupported param combos.
- Time-boxed and iteration-limited; stops on first failure for actionable logs.
"""
from __future__ import annotations
import argparse, math, sys, time, random
import numpy as np


def main() -> int:
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

    HAS = {
        "sample": hasattr(S, "sample"),
        "make_rng": hasattr(S, "make_rng"),
        "categorical": hasattr(S, "categorical"),
        "mixture": hasattr(S, "mixture"),
        "survival_times": hasattr(S, "survival_times"),
        "dirichlet": hasattr(S, "dirichlet"),
    }
    make_rng = S.make_rng if HAS["make_rng"] else (lambda seed=None: np.random.default_rng(seed))

    while checked < args.iterations and (time.time() - t0) < args.timeout_sec:
        checked += 1
        rng = make_rng(random.randrange(0, 2**32 - 1))
        case = {"op": None, "dist": None, "params": None, "size": None}

        try:
            # Pick an available op
            ops = [k for k, v in HAS.items() if v and k in ("sample", "categorical", "mixture", "survival_times", "dirichlet")]
            op = random.choice(ops)
            case["op"] = op

            if op == "sample":
                dist = random.choice(["poisson", "lognormal", "negbin", "normal"])
                n = random.randint(1, 5000)
                case.update(dist=dist, size=n)

                if dist == "poisson":
                    lam = 10.0 ** random.uniform(-2, 1.2)  # ~0.01..15
                    case["params"] = {"lam": lam}
                    x = S.sample("poisson", n, rng=rng, lam=lam)
                    assert isinstance(x, np.ndarray) and x.shape == (n,), f"poisson shape {x.shape} != ({n},)"
                    assert np.isfinite(x).all(), "poisson non-finite values"
                    assert (x >= 0).all(), "poisson negative values"
                    assert np.issubdtype(x.dtype, np.integer), f"poisson dtype {x.dtype} not integer"

                elif dist == "lognormal":
                    # quantile-based parameterization; library should raise on ill-conditioned pairs
                    x1, x2 = sorted(10.0 ** np.random.uniform(-1, 2, size=2))  # 0.1..100
                    p1, p2 = sorted(np.random.uniform(0.1, 0.99, size=2))
                    par = {"from_quantiles": True, "x1": float(x1), "p1": float(p1), "x2": float(x2), "p2": float(p2)}
                    case["params"] = par
                    x = S.sample("lognormal", n, rng=rng, **par)
                    assert x.shape == (n,), f"lognormal shape {x.shape} != ({n},)"
                    assert np.isfinite(x).all(), "lognormal non-finite"
                    assert (x > 0).all(), "lognormal non-positive draw"

                elif dist == "negbin":
                    mu = 10.0 ** random.uniform(-2, 1.3)  # ~0.01..20
                    k = 10.0 ** random.uniform(-2, 1.0)   # >0
                    par = {"mean": float(mu), "k": float(k)}
                    case["params"] = par
                    x = S.sample("negbin", n, rng=rng, **par)
                    assert x.shape == (n,), f"negbin shape {x.shape} != ({n},)"
                    assert np.isfinite(x).all(), "negbin non-finite"
                    assert (x >= 0).all(), "negbin negative values"
                    assert np.issubdtype(x.dtype, np.integer), f"negbin dtype {x.dtype} not integer"

                elif dist == "normal":
                    loc = random.uniform(-5, 5)
                    scale = 10.0 ** random.uniform(-2, 1)  # >0
                    par = {"loc": float(loc), "scale": float(scale)}
                    case["params"] = par
                    x = S.sample("normal", n, rng=rng, **par)
                    assert x.shape == (n,), f"normal shape {x.shape} != ({n},)"
                    assert np.isfinite(x).all(), "normal non-finite"

            elif op == "categorical":
                k = random.randint(2, 6)
                w = np.abs(np.random.default_rng().normal(size=k)); w = w / w.sum()
                size = random.randint(1, 4000)
                labels = [f"L{i}" for i in range(k)]
                case.update(size=size, params={"weights": w.tolist(), "labels": labels})
                lbls = S.categorical(w, size=size, rng=rng, labels=labels)
                assert len(lbls) == size, f"categorical size {len(lbls)} != {size}"
                assert set(lbls).issubset(set(labels)), "categorical labels outside provided set"

            elif op == "mixture":
                comps = [("normal", {"loc": 0.0, "scale": 1.0}),
                         ("normal", {"loc": random.uniform(2, 12), "scale": 1.0})]
                weights = [0.5, 0.5]
                size = random.randint(10, 10000)
                case.update(size=size, params={"components": comps, "weights": weights})
                x = S.mixture(components=comps, weights=weights, size=size, rng=rng)
                assert isinstance(x, np.ndarray) and x.shape == (size,), f"mixture shape {x.shape} != ({size},)"
                assert np.isfinite(x).all(), "mixture non-finite"

            elif op == "survival_times":
                lam = 10.0 ** random.uniform(-3, -0.3)  # hazard ~0.001..0.5
                size = random.randint(10, 5000)
                censor_at = random.choice([None, random.uniform(1.0, 20.0)])
                par = {"lam": float(lam), "censor_at": censor_at}
                case.update(size=size, params=par, dist="exponential")
                t, e = S.survival_times("exponential", size, rng=rng, **par)
                assert len(t) == size and len(e) == size, f"survival lengths {(len(t), len(e))} != {size}"
                assert np.isfinite(t).all(), "survival times non-finite"
                assert (t >= 0).all(), "survival times negative"
                ev = set(np.unique(e))
                assert ev.issubset({0, 1}), f"survival events invalid set {ev}"
                if censor_at is not None:
                    assert (t <= censor_at + 1e-9).all(), "survival t > censor_at"

            elif op == "dirichlet":
                k = random.randint(2, 6)
                alpha = 10.0 ** np.random.uniform(-2, 1, size=k)  # >0
                case.update(params={"alpha": alpha.tolist()}, dist="dirichlet", size=1)
                x = S.dirichlet(alpha, rng=rng)
                x = np.asarray(x)
                assert np.isfinite(x).all(), "dirichlet non-finite"
                assert (x >= 0).all(), "dirichlet negative"
                if x.ndim == 1:
                    s = float(np.sum(x))
                    assert math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6), f"dirichlet sum {s}"
                elif x.ndim == 2:
                    row_sums = np.sum(x, axis=1)
                    assert np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6), f"dirichlet row sums {row_sums}"
                else:
                    raise AssertionError(f"dirichlet unexpected ndim {x.ndim}")

        except (ValueError, NotImplementedError):
            # Invalid/unsupported cases are fine: skip and continue.
            continue
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] case={case} :: {e}", file=sys.stderr)
            break  # stop on first failure so CI log is focused/actionable
        except Exception as e:
            failures += 1
            print(f"[CRASH] case={case} :: {type(e).__name__}: {e}", file=sys.stderr)
            break

    print(f"Fuzzed {checked} cases; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
