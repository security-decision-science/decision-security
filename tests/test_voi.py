import numpy as np
import pytest

from decision_security.voi import evpi, select_controls_by_roi


def test_evpi_is_nonnegative_random():
    rng = np.random.default_rng(42)
    for _ in range(50):
        L = rng.lognormal(10, 2, size=(200, 4))
        assert evpi(L) >= 0.0


def test_evpi_known_value():
    # Two equally likely states, two decisions.
    # E[L(d1)] = 50, E[L(d2)] = 50 -> min_d E[L] = 50
    # E[min_d L] = (0 + 0) / 2 = 0 -> EVPI = 50
    L = np.array([[0.0, 100.0], [100.0, 0.0]])
    assert evpi(L) == pytest.approx(50.0)


def test_evpi_zero_when_one_decision_dominates():
    # Decision 1 is best in every scenario: perfect information changes nothing.
    L = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    assert evpi(L) == pytest.approx(0.0)


def test_select_controls_by_roi_respects_budget():
    idx, spent, gained = select_controls_by_roi(
        deltas=[100, 90, 50], costs=[100, 50, 60], budget=110
    )
    assert spent <= 110
    assert 1 in idx  # highest ratio (1.8) always fits first
