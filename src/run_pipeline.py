"""End-to-end churn pipeline: train -> evaluate -> explain -> recommend.

Run:  python -m src.run_pipeline
"""
from __future__ import annotations

import os

import pandas as pd

from .data import load_raw, split
from .decision import optimize_threshold
from .explain import compute_shap, global_drivers, plain_english_recommendations
from .logging_utils import get_logger, log_timing
from .model import train_and_evaluate

log = get_logger(__name__)

PLOT_PATH = os.path.join("data", "shap_drivers.png")


def run(threshold: float = 0.5) -> dict:
    df = load_raw()
    ds = split(df)
    log.info("Training on %d rows (churn rate %.1f%%)", len(df), 100 * df["churn"].mean())
    with log_timing(log, "train + evaluate (LogReg, XGBoost)"):
        trained = train_and_evaluate(ds, threshold=threshold)

    results_df = pd.DataFrame([r.as_row() for r in trained["results"]])

    xgb = trained["fitted"]["XGBoost"]
    explanation = compute_shap(xgb, ds.X_train)
    drivers = global_drivers(explanation)
    recommendations = plain_english_recommendations(drivers)

    # Turn probabilities into a profit-maximising "who to contact" decision.
    proba_test = xgb.predict_proba(ds.X_test)[:, 1]
    decision = optimize_threshold(ds.y_test, proba_test)
    log.info("Profit-optimal threshold=%.2f, contact %d customers",
             decision.threshold, decision.n_contacted)

    return {
        "churn_rate": float(df["churn"].mean()),
        "results": results_df,
        "drivers": drivers,
        "recommendations": recommendations,
        "explanation": explanation,
        "decision": decision,
    }


def save_plot(drivers: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = drivers.head(10).iloc[::-1]
        colors = ["#d6336c" if "+" in e else "#1c7ed6" for e in top["effect"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top["feature"], top["importance"], color=colors)
        ax.set_title("Top churn drivers (mean |SHAP|)  — red: +churn, blue: -churn")
        ax.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        os.makedirs("data", exist_ok=True)
        fig.savefig(PLOT_PATH, dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


def main() -> None:
    out = run()
    print(f"\nChurn rate: {out['churn_rate']:.1%}\n")
    print("=== Model performance (test set) ===")
    print(out["results"].to_string(index=False))
    print("\n=== Top churn drivers (SHAP) ===")
    print(out["drivers"].head(10).to_string(index=False))
    print("\n=== Recommended retention actions ===")
    for i, rec in enumerate(out["recommendations"], 1):
        print(f"  {i}. {rec}")
    d = out["decision"]
    print("\n=== Campaign economics (CLV=1000, offer=50, retain_rate=30%) ===")
    print(f"  Profit-optimal threshold: {d.threshold:.2f}")
    print(f"  Customers to contact:     {d.n_contacted}")
    print(f"  Expected campaign profit: {d.expected_profit:,.0f}")
    save_plot(out["drivers"])
    print(f"\nDriver plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
