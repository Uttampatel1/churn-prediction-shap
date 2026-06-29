import pytest

from src import service
from src.generate_data import generate_dataset
from src.service import ChurnScorer, create_app


@pytest.fixture(scope="module")
def scorer():
    # Small, fixed dataset + fixed threshold -> fast and deterministic.
    df = generate_dataset(n=2500, seed=11)
    return ChurnScorer.from_training(df=df, threshold=0.5)


def _high_risk():
    return {
        "customer_id": "hi",
        "tenure_months": 1, "monthly_charges": 95.0, "support_tickets": 6,
        "num_products": 1, "is_senior": 1, "auto_pay": 0, "premium_support": 0,
        "contract": "Month-to-month", "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
    }


def _low_risk():
    return {
        "customer_id": "lo",
        "tenure_months": 60, "monthly_charges": 25.0, "support_tickets": 0,
        "num_products": 3, "is_senior": 0, "auto_pay": 1, "premium_support": 1,
        "contract": "Two year", "payment_method": "Credit card",
        "internet_service": "DSL",
    }


def test_empty_batch_returns_empty(scorer):
    assert scorer.score_batch([]) == []


def test_score_batch_shape_and_keys(scorer):
    out = scorer.score_batch([_high_risk(), _low_risk()])
    assert len(out) == 2
    r = out[0]
    assert {"customer_id", "churn_probability", "decision", "risk_band",
            "top_reasons", "recommended_action"} <= r.keys()
    assert 0.0 <= r["churn_probability"] <= 1.0
    assert r["decision"] in {"contact", "monitor"}
    assert len(r["top_reasons"]) == 4


def test_high_risk_scores_above_low_risk(scorer):
    hi, lo = scorer.score_batch([_high_risk(), _low_risk()])
    assert hi["churn_probability"] > lo["churn_probability"]
    assert hi["risk_band"] == "high"
    # a churn-driving customer should get an actionable retention recommendation
    assert hi["recommended_action"] is not None


def test_decision_follows_threshold(scorer):
    out = scorer.score_batch([_high_risk()])[0]
    expected = "contact" if out["churn_probability"] >= scorer.threshold else "monitor"
    assert out["decision"] == expected


# --- API tests -------------------------------------------------------------

@pytest.fixture
def client(scorer, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(service, "get_scorer", lambda: scorer)
    return TestClient(create_app())


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_endpoint(client):
    resp = client.post("/score", json={"customers": [_high_risk(), _low_risk()]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_scored"] == 2
    assert 0 <= body["n_to_contact"] <= 2
    assert len(body["results"]) == 2


def test_score_endpoint_validates_input(client):
    bad = _high_risk()
    bad["is_senior"] = 5  # out of [0, 1] range
    resp = client.post("/score", json={"customers": [bad]})
    assert resp.status_code == 422


def test_score_endpoint_rejects_empty_batch(client):
    resp = client.post("/score", json={"customers": []})
    assert resp.status_code == 422
