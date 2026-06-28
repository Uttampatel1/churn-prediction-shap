from src.data import split
from src.explain import (
    compute_shap,
    explain_customer,
    global_drivers,
    plain_english_recommendations,
)
from src.generate_data import generate_dataset
from src.model import build_xgboost


def _fitted():
    ds = split(generate_dataset(n=2500, seed=11))
    pipe = build_xgboost().fit(ds.X_train, ds.y_train)
    return pipe, ds


def test_shap_recovers_known_drivers():
    pipe, ds = _fitted()
    drivers = global_drivers(compute_shap(pipe, ds.X_train))
    top = set(drivers["feature"].head(5))
    # tenure and contract type are the strongest true drivers.
    assert "tenure_months" in top
    assert any("contract_Month-to-month" == f for f in drivers["feature"])

    tenure_effect = drivers.loc[drivers["feature"] == "tenure_months", "effect"].iloc[0]
    assert tenure_effect == "-churn"  # longer tenure reduces churn


def test_recommendations_nonempty():
    pipe, ds = _fitted()
    recs = plain_english_recommendations(global_drivers(compute_shap(pipe, ds.X_train)))
    assert len(recs) >= 3
    assert all(isinstance(r, str) for r in recs)


def test_explain_customer_reason_codes():
    pipe, ds = _fitted()
    out = explain_customer(pipe, ds.X_test.head(1))
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert len(out["top_reasons"]) == 4
    assert {"feature", "shap", "effect"} <= out["top_reasons"][0].keys()
