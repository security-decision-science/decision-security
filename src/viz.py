import numpy as np
import matplotlib.pyplot as plt


def plot_loss_distribution(losses, bins=50, ax=None):
    arr = np.asarray(losses, dtype=float)
    ax = ax or plt.gca()
    ax.hist(arr, bins=bins)
    ax.set_xlabel("Aggregate loss"); ax.set_ylabel("Count")
    ax.set_title("Loss distribution")
    return ax

def plot_risk_bands(losses, ax=None):
    arr = np.asarray(losses, dtype=float)
    p50, p90, p95 = np.quantile(arr, [0.5, 0.9, 0.95])
    ax = ax or plt.gca()
    ax.hist(arr, bins=50)
    for q, v in [("P50", p50), ("P90", p90), ("P95", p95)]:
        ax.axvline(v, linestyle="--")
        ax.text(v, ax.get_ylim()[1]*0.9, q, rotation=90, va="top")
    ax.set_xlabel("Aggregate loss"); ax.set_title("Risk bands")
    return ax

def plot_km(times, surv, ax=None):
    t = np.asarray(times, dtype=float)
    s = np.asarray(surv, dtype=float)
    ax = ax or plt.gca()
    ax.step(t, s, where="post")
    ax.set_xlabel("Time"); ax.set_ylabel("Survival S(t)")
    ax.set_title("Kaplan–Meier")
    return ax