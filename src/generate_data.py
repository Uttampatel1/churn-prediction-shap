"""Generate a synthetic SaaS/telecom-style churn dataset.

Churn is generated from an interpretable log-odds model of the features, so the
true drivers are known and a good model + SHAP should recover them. This makes
the explainability story verifiable rather than hand-wavy.

Run:  python -m src.generate_data
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Credit card", "Bank transfer"]
INTERNET = ["DSL", "Fiber optic", "None"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly_charges = np.round(rng.normal(70, 25, n).clip(20, 140), 2)
    support_tickets = rng.poisson(1.3, n)
    num_products = rng.integers(1, 6, n)
    senior = rng.binomial(1, 0.18, n)
    auto_pay = rng.binomial(1, 0.55, n)
    premium_support = rng.binomial(1, 0.35, n)

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENTS, n, p=[0.4, 0.35, 0.25])
    internet = rng.choice(INTERNET, n, p=[0.4, 0.45, 0.15])

    # Interpretable log-odds of churn.
    log_odds = (
        -0.5
        - 0.045 * tenure                       # longer tenure -> sticky
        + 0.018 * (monthly_charges - 70)       # pricier -> more churn
        + 0.35 * support_tickets               # friction -> churn
        - 0.30 * (num_products - 1)            # more products -> sticky
        + 0.45 * senior
        - 0.60 * auto_pay
        - 0.80 * premium_support
        + np.where(contract == "Month-to-month", 1.3, 0.0)
        + np.where(contract == "Two year", -1.1, 0.0)
        + np.where(payment == "Electronic check", 0.6, 0.0)
        + np.where(internet == "Fiber optic", 0.4, 0.0)
    )
    prob = _sigmoid(log_odds + rng.normal(0, 0.4, n))
    churn = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "support_tickets": support_tickets,
            "num_products": num_products,
            "is_senior": senior,
            "auto_pay": auto_pay,
            "premium_support": premium_support,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "churn": churn,
        }
    )


def main(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    df = generate_dataset()
    path = os.path.join(data_dir, "churn.csv")
    df.to_csv(path, index=False)
    rate = df["churn"].mean()
    print(f"Wrote {len(df):,} rows to {path}  (churn rate {rate:.1%})")
    return path


if __name__ == "__main__":
    main()
