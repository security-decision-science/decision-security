import numpy as np

def km_estimator(times, events):
    """
    Kaplan–Meier survival curve.
    times: event or censor times
    events: 1 if event observed, 0 if right-censored
    Returns (unique_times, survival_probabilities)
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t, kind="mergesort")
    t = t[order]; e = e[order]
    uniq = np.unique(t)
    n_at_risk = len(t)
    s = 1.0
    surv_t, surv = [], []
    for ut in uniq:
        mask = t == ut
        d = int(e[mask].sum())
        if n_at_risk > 0:
            s *= (1.0 - d / n_at_risk)
        surv_t.append(float(ut))
        surv.append(float(s))
        n_at_risk -= int(mask.sum())
    return np.asarray(surv_t), np.asarray(surv)

def nelson_aalen(times, events):
    """
    Nelson–Aalen cumulative hazard H(t).
    Returns (unique_times, cumulative_hazard)
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t, kind="mergesort")
    t = t[order]; e = e[order]
    uniq = np.unique(t)
    n_at_risk = len(t)
    H = 0.0
    haz_t, cumhaz = [], []
    for ut in uniq:
        mask = t == ut
        d = int(e[mask].sum())
        if n_at_risk > 0:
            H += d / n_at_risk
        haz_t.append(float(ut)); cumhaz.append(float(H))
        n_at_risk -= int(mask.sum())
    return np.asarray(haz_t), np.asarray(cumhaz)