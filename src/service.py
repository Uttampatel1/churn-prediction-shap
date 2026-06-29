"""FastAPI batch-scoring microservice — make the churn model *deployable*.

A model that lives in a notebook can't drive a retention campaign. This service
wraps the trained XGBoost pipeline behind an HTTP API: POST a batch of customers
and get back, for each one, a churn probability, a profit-based **contact / monitor
decision**, the **SHAP reason codes** behind the score, and a concrete **retention
action**. That's the difference between "we built a model" and "ops can call it
every night and act on the output".

Run:  uvicorn src.service:app --reload      (or: python -m src.service)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .data import CATEGORICAL, NUMERIC, feature_names, load_raw, split
from .decision import optimize_threshold
from .explain import RECOMMENDATIONS
from .model import build_xgboost


class ChurnScorer:
    """A fitted churn pipeline plus a profit-based decision threshold.

    Computes SHAP reason codes for a whole batch in one pass (one explainer call),
    which is far cheaper than per-row explanation when scoring nightly batches.
    """

    def __init__(self, pipe, threshold: float = 0.5):
        self.pipe = pipe
        self.threshold = float(threshold)
        self._prep = pipe.named_steps["prep"]
        self._clf = pipe.named_steps["clf"]
        self._names = feature_names(self._prep)

    @classmethod
    def from_training(cls, df: pd.DataFrame | None = None, threshold: float | None = None) -> "ChurnScorer":
        """Train on the (synthetic) churn data and pick a profit-optimal threshold."""
        if df is None:
            df = load_raw()
        ds = split(df)
        pipe = build_xgboost()
        pipe.fit(ds.X_train, ds.y_train)
        if threshold is None:
            proba = pipe.predict_proba(ds.X_test)[:, 1]
            threshold = optimize_threshold(ds.y_test, proba).threshold
        return cls(pipe, threshold)

    def _risk_band(self, p: float) -> str:
        if p >= 0.66:
            return "high"
        if p >= 0.33:
            return "medium"
        return "low"

    def _action_for(self, reasons: list[dict]) -> str | None:
        """First retention action whose driver is pushing this customer *toward* churn."""
        for r in reasons:
            if r["effect"] == "+churn" and r["feature"] in RECOMMENDATIONS:
                return RECOMMENDATIONS[r["feature"]]
        return None

    def score_batch(self, records: list[dict], top_n: int = 4) -> list[dict]:
        if not records:
            return []
        X = pd.DataFrame(records)[NUMERIC + CATEGORICAL]
        proba = self.pipe.predict_proba(X)[:, 1]

        import shap

        X_t = self._prep.transform(X)
        sv = shap.TreeExplainer(self._clf).shap_values(X_t)
        if isinstance(sv, list):  # some shap versions return a per-class list
            sv = sv[-1]
        sv = np.asarray(sv)

        out: list[dict] = []
        for i, rec in enumerate(records):
            order = np.argsort(-np.abs(sv[i]))[:top_n]
            reasons = [
                {
                    "feature": self._names[j],
                    "shap": round(float(sv[i, j]), 4),
                    "effect": "+churn" if sv[i, j] > 0 else "-churn",
                }
                for j in order
            ]
            p = float(proba[i])
            out.append({
                "customer_id": rec.get("customer_id"),
                "churn_probability": round(p, 4),
                "decision": "contact" if p >= self.threshold else "monitor",
                "risk_band": self._risk_band(p),
                "top_reasons": reasons,
                "recommended_action": self._action_for(reasons),
            })
        return out


# --- API layer -------------------------------------------------------------

class CustomerRecord(BaseModel):
    customer_id: str | None = None
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    num_products: int = Field(ge=0)
    is_senior: int = Field(ge=0, le=1)
    auto_pay: int = Field(ge=0, le=1)
    premium_support: int = Field(ge=0, le=1)
    contract: str
    payment_method: str
    internet_service: str


class ScoreRequest(BaseModel):
    customers: list[CustomerRecord] = Field(min_length=1)


@lru_cache(maxsize=1)
def get_scorer() -> ChurnScorer:
    """Train once, reuse across requests (cached for the process lifetime)."""
    return ChurnScorer.from_training()


def create_app():
    from fastapi import FastAPI

    app = FastAPI(
        title="Churn Batch-Scoring API",
        description="Score a batch of customers: probability + SHAP reasons + retention action.",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict:
        scorer = get_scorer()
        return {"status": "ok", "threshold": scorer.threshold}

    @app.post("/score")
    def score(req: ScoreRequest) -> dict:
        scorer = get_scorer()
        records = [c.model_dump() for c in req.customers]
        results = scorer.score_batch(records)
        return {
            "threshold": scorer.threshold,
            "n_scored": len(results),
            "n_to_contact": sum(r["decision"] == "contact" for r in results),
            "results": results,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
