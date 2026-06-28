"""Churn explainability dashboard.

Shows the global churn drivers and lets you score a single customer with
SHAP-based reason codes — the "why" behind each prediction.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import CATEGORICAL, NUMERIC, load_raw, split
from src.explain import compute_shap, explain_customer, global_drivers, plain_english_recommendations
from src.model import build_xgboost, evaluate

st.set_page_config(page_title="Churn Explainability", page_icon="🧠", layout="wide")


@st.cache_resource
def _train():
    df = load_raw()
    ds = split(df)
    pipe = build_xgboost().fit(ds.X_train, ds.y_train)
    res = evaluate("XGBoost", pipe, ds)
    drivers = global_drivers(compute_shap(pipe, ds.X_train))
    return df, ds, pipe, res, drivers


df, ds, pipe, res, drivers = _train()

st.title("🧠 Customer Churn — Drivers & Actions")
c1, c2, c3 = st.columns(3)
c1.metric("Churn rate", f"{df['churn'].mean():.1%}")
c2.metric("ROC-AUC", f"{res.roc_auc:.3f}")
c3.metric("PR-AUC", f"{res.pr_auc:.3f}")

st.subheader("What drives churn (global SHAP importance)")
top = drivers.head(10).iloc[::-1]
fig = px.bar(
    top, x="importance", y="feature", orientation="h", color="effect",
    color_discrete_map={"+churn": "#d6336c", "-churn": "#1c7ed6", "~": "#adb5bd"},
)
fig.update_layout(height=420, yaxis_title="", xaxis_title="mean |SHAP|")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recommended retention actions")
for rec in plain_english_recommendations(drivers):
    st.write(f"- {rec}")

st.divider()
st.subheader("Score a single customer")
cols = st.columns(4)
inp = {
    "tenure_months": cols[0].slider("Tenure (months)", 1, 72, 6),
    "monthly_charges": cols[1].slider("Monthly charges", 20, 140, 95),
    "support_tickets": cols[2].slider("Support tickets", 0, 10, 3),
    "num_products": cols[3].slider("# products", 1, 5, 1),
    "is_senior": cols[0].selectbox("Senior?", [0, 1]),
    "auto_pay": cols[1].selectbox("Auto-pay?", [0, 1]),
    "premium_support": cols[2].selectbox("Premium support?", [0, 1]),
    "contract": cols[3].selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "payment_method": cols[0].selectbox("Payment", ["Electronic check", "Credit card", "Bank transfer"]),
    "internet_service": cols[1].selectbox("Internet", ["DSL", "Fiber optic", "None"]),
}
customer = pd.DataFrame([inp])[NUMERIC + CATEGORICAL]
out = explain_customer(pipe, customer)
st.metric("Predicted churn probability", f"{out['churn_probability']:.1%}")
st.write("**Top reasons:**")
st.table(pd.DataFrame(out["top_reasons"]))
