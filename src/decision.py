"""From churn probabilities to a *decision*: who to contact, and the threshold.

A ROC-AUC of 0.83 is nice, but the business question is "who do we call, and is
it worth it?". That depends on economics, not on 0.5:

* contacting a real churner who we then save is worth ``retain_rate * clv`` minus
  the ``offer_cost`` of the retention incentive (a **true positive**);
* contacting someone who wasn't going to leave just wastes ``offer_cost`` (a
  **false positive**);
* missing a churner loses their ``clv`` (a **false negative**).

This module sweeps the probability threshold and picks the one that maximises
expected campaign profit — the number you'd actually defend to a CFO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdResult:
    threshold: float
    expected_profit: float
    n_contacted: int
    thresholds: np.ndarray
    profit_curve: np.ndarray


def expected_profit(
    y_true,
    proba,
    threshold: float,
    clv: float = 1000.0,
    offer_cost: float = 50.0,
    retain_rate: float = 0.30,
) -> float:
    """Expected profit of running the retention campaign at ``threshold``.

    Profit is measured against doing nothing for the *contacted* group, so a
    customer we never call contributes nothing either way.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    contacted = proba >= threshold

    # Value per contacted customer, by whether they were truly going to churn.
    tp = int(np.sum(contacted & (y_true == 1)))
    fp = int(np.sum(contacted & (y_true == 0)))
    value_tp = retain_rate * clv - offer_cost
    value_fp = -offer_cost
    return float(tp * value_tp + fp * value_fp)


def optimize_threshold(
    y_true,
    proba,
    clv: float = 1000.0,
    offer_cost: float = 50.0,
    retain_rate: float = 0.30,
    n_grid: int = 101,
) -> ThresholdResult:
    """Grid-search the threshold that maximises expected campaign profit.

    "Run no campaign at all" (profit 0) is always one of the options, so a
    money-losing offer correctly yields a threshold above every probability and
    contacts no one.
    """
    # The grid runs just past 1.0 so the top end means "contact nobody".
    thresholds = np.append(np.linspace(0.0, 1.0, n_grid), 1.0 + 1e-9)
    profits = np.array([
        expected_profit(y_true, proba, t, clv, offer_cost, retain_rate)
        for t in thresholds
    ])
    best_i = int(np.argmax(profits))
    best_t = float(thresholds[best_i])
    n_contacted = int(np.sum(np.asarray(proba, dtype=float) >= best_t))
    return ThresholdResult(
        threshold=best_t,
        expected_profit=float(profits[best_i]),
        n_contacted=n_contacted,
        thresholds=thresholds,
        profit_curve=profits,
    )
