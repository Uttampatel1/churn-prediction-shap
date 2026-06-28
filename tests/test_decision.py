from __future__ import annotations

import numpy as np

from src.decision import expected_profit, optimize_threshold


def _toy():
    # 200 customers; the model's probability is correlated with the true label.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    proba = np.clip(0.5 * y + rng.normal(0, 0.2, 400) + 0.25, 0, 1)
    return y, proba


def test_optimize_returns_valid_threshold():
    y, proba = _toy()
    res = optimize_threshold(y, proba, clv=1000, offer_cost=50, retain_rate=0.3)
    assert 0.0 <= res.threshold <= 1.0
    assert res.profit_curve.shape == res.thresholds.shape


def test_optimal_profit_beats_naive_half():
    y, proba = _toy()
    res = optimize_threshold(y, proba, clv=1000, offer_cost=50, retain_rate=0.3)
    half = expected_profit(y, proba, 0.5, clv=1000, offer_cost=50, retain_rate=0.3)
    assert res.expected_profit >= half


def test_expensive_offer_contacts_fewer_people():
    y, proba = _toy()
    cheap = optimize_threshold(y, proba, clv=1000, offer_cost=20, retain_rate=0.5)
    pricey = optimize_threshold(y, proba, clv=1000, offer_cost=250, retain_rate=0.5)
    # A worse-value offer should make the model pickier (higher bar to contact).
    assert pricey.threshold >= cheap.threshold
    assert pricey.n_contacted <= cheap.n_contacted


def test_worthless_offer_contacts_nobody():
    y, proba = _toy()
    # If saving a customer is worth less than the offer costs, contact no one.
    res = optimize_threshold(y, proba, clv=100, offer_cost=100, retain_rate=0.1)
    assert res.n_contacted == 0
