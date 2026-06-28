"""Model training and evaluation.

Trains a Logistic Regression baseline and an XGBoost classifier, then evaluates
both with metrics that matter for churn: ROC-AUC, PR-AUC (better under class
imbalance), and precision/recall/F1 at a chosen threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .data import Dataset, build_preprocessor


@dataclass
class EvalResult:
    name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion: list[list[int]]

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "ROC_AUC": round(self.roc_auc, 4),
            "PR_AUC": round(self.pr_auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
        }


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def build_xgboost() -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=False)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(name: str, pipe: Pipeline, ds: Dataset, threshold: float = 0.5) -> EvalResult:
    proba = pipe.predict_proba(ds.X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return EvalResult(
        name=name,
        roc_auc=roc_auc_score(ds.y_test, proba),
        pr_auc=average_precision_score(ds.y_test, proba),
        precision=precision_score(ds.y_test, preds, zero_division=0),
        recall=recall_score(ds.y_test, preds, zero_division=0),
        f1=f1_score(ds.y_test, preds, zero_division=0),
        threshold=threshold,
        confusion=confusion_matrix(ds.y_test, preds).tolist(),
    )


def train_and_evaluate(ds: Dataset, threshold: float = 0.5) -> dict[str, Any]:
    models = {"LogisticRegression": build_logreg(), "XGBoost": build_xgboost()}
    fitted: dict[str, Pipeline] = {}
    results: list[EvalResult] = []
    for name, pipe in models.items():
        pipe.fit(ds.X_train, ds.y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, ds, threshold))
    return {"fitted": fitted, "results": results}
