import numpy as np

def evpi(loss_matrix):
    """
    Expected Value of Perfect Information.
    loss_matrix: shape (S, D) — S scenarios (or posterior draws), D decisions.
    EVPI = E[min_d L(d,θ)] - min_d E[L(d,θ)]
    """
    L = np.asarray(loss_matrix, dtype=float)
    term1 = float(L.min(axis=1).mean())
    term2 = float(L.mean(axis=0).min())
    return term1 - term2

def select_controls_by_roi(deltas, costs, budget):
    """
    Greedy selection by Δrisk/cost ratio. Returns (indices, total_cost, total_delta).
    deltas: positive numbers = risk reduction achieved by control
    costs:  cost per control
    budget: total budget available
    """
    d = np.asarray(list(deltas), dtype=float)
    c = np.asarray(list(costs), dtype=float)
    ratio = d / np.maximum(c, 1e-12)
    order = np.argsort(-ratio)
    chosen, spent, gained = [], 0.0, 0.0
    for i in order:
        if spent + c[i] <= budget:
            chosen.append(int(i))
            spent += float(c[i])
            gained += float(d[i])
    return chosen, spent, gained