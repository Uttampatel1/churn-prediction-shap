"""Data loading and preprocessing for the churn models."""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
ID_COL = "customer_id"
NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "num_products",
    "is_senior",
    "auto_pay",
    "premium_support",
]
CATEGORICAL = ["contract", "payment_method", "internet_service"]


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw(path: str = "data/churn.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        from .generate_data import main

        main(os.path.dirname(path) or ".")
    return pd.read_csv(path)


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> Dataset:
    features = NUMERIC + CATEGORICAL
    X = df[features].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(X_train, X_test, y_train, y_test)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot encode categoricals; optionally scale numerics (for linear models)."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable names after one-hot expansion."""
    names = list(NUMERIC)
    ohe = preprocessor.named_transformers_["cat"]
    names.extend(ohe.get_feature_names_out(CATEGORICAL).tolist())
    return names
