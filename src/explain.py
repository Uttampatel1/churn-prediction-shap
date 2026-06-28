"""SHAP-based explainability + plain-English retention recommendations.

The point of churn modeling isn't the score — it's knowing *why* customers leave
and *what to do about it*. This module turns SHAP values into a ranked list of
drivers (with direction) and maps the top drivers to concrete retention actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import feature_names

# Maps a raw feature to a (driver-increases-churn, action) recommendation.
RECOMMENDATIONS: dict[str, str] = {
    "tenure_months": "Most churn happens early — invest in 30/60/90-day onboarding and check-ins.",
    "monthly_charges": "High bills drive churn — offer right-sized plans or loyalty discounts to high-spend accounts.",
    "support_tickets": "Support friction predicts churn — proactively follow up after every 2+ ticket interaction.",
    "num_products": "Single-product users churn more — drive cross-sell to deepen account stickiness.",
    "premium_support": "Premium support reduces churn — bundle or trial it for at-risk customers.",
    "auto_pay": "Auto-pay reduces churn — incentivize enrollment (e.g., small discount).",
    "is_senior": "Senior customers churn more — tailor onboarding and support channels to them.",
    "contract_Month-to-month": "Month-to-month customers churn most — incentivize annual contracts.",
    "contract_Two year": "Long contracts retain — promote 1/2-year plans at renewal.",
    "payment_method_Electronic check": "Electronic-check payers churn more — nudge toward card/bank auto-pay.",
    "internet_service_Fiber optic": "Fiber customers churn more (price-sensitive) — review fiber value/pricing.",
}


@dataclass
class ShapExplanation:
    feature_names: list[str]
    shap_values: np.ndarray  # (n_samples, n_features)
    X_transformed: np.ndarray


def compute_shap(pipe: Pipeline, X: pd.DataFrame, max_samples: int = 1000) -> ShapExplanation:
    import shap

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = feature_names(prep)

    X_t = prep.transform(X)
    if len(X_t) > max_samples:
        idx = np.random.default_rng(0).choice(len(X_t), max_samples, replace=False)
        X_t = X_t[idx]

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):  # some versions return per-class lists
        values = values[-1]
    return ShapExplanation(feature_names=names, shap_values=np.asarray(values), X_transformed=X_t)


def global_drivers(explanation: ShapExplanation) -> pd.DataFrame:
    """Rank features by mean |SHAP|, with the direction of their effect on churn."""
    shap_vals = explanation.shap_values
    X = explanation.X_transformed
    mean_abs = np.abs(shap_vals).mean(axis=0)

    directions = []
    for j in range(shap_vals.shape[1]):
        col = X[:, j]
        sv = shap_vals[:, j]
        if np.std(col) < 1e-9 or np.std(sv) < 1e-9:
            directions.append("~")
        else:
            corr = np.corrcoef(col, sv)[0, 1]
            directions.append("+churn" if corr > 0 else "-churn")

    return (
        pd.DataFrame(
            {
                "feature": explanation.feature_names,
                "importance": mean_abs,
                "effect": directions,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plain_english_recommendations(drivers: pd.DataFrame, top_n: int = 6) -> list[str]:
    out: list[str] = []
    for feat in drivers["feature"].head(top_n):
        if feat in RECOMMENDATIONS:
            out.append(RECOMMENDATIONS[feat])
    # De-duplicate while preserving order.
    seen: set[str] = set()
    return [r for r in out if not (r in seen or seen.add(r))]


def explain_customer(pipe: Pipeline, customer: pd.DataFrame, top_n: int = 4) -> dict:
    """Per-customer reason codes: top features pushing this prediction."""
    import shap

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = feature_names(prep)
    x_t = prep.transform(customer)
    sv = shap.TreeExplainer(clf).shap_values(x_t)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)[0]
    proba = float(pipe.predict_proba(customer)[0, 1])
    order = np.argsort(-np.abs(sv))[:top_n]
    reasons = [
        {"feature": names[i], "shap": round(float(sv[i]), 4),
         "effect": "+churn" if sv[i] > 0 else "-churn"}
        for i in order
    ]
    return {"churn_probability": round(proba, 4), "top_reasons": reasons}
