# 🧠 Customer Churn Prediction + Explainability (SHAP)

**Business question:** *Which customers are about to leave, **why**, and what should we do about it?* A churn score alone doesn't retain anyone — teams need the drivers and the actions.

This project predicts churn on a SaaS/telecom-style dataset and uses **SHAP** to turn the model into a ranked list of drivers and **plain-English retention recommendations**, plus per-customer reason codes.

---

## Key findings (what the business should act on)

**Top 3 churn drivers** (validated with SHAP, directions correct):

1. **Short tenure** — churn is concentrated in the first months *(↓ churn as tenure grows)*.
2. **Month-to-month contracts** — by far the biggest *increaser* of churn risk.
3. **High monthly charges** — price sensitivity drives departures.

Other significant drivers: few products held, frequent support tickets, no premium support, no auto-pay, electronic-check payment, fiber-optic service.

**Recommended retention actions** (auto-generated from the drivers):

- Invest in **30/60/90-day onboarding** — most churn happens early.
- **Incentivize annual contracts** for month-to-month customers.
- **Right-size plans / loyalty discounts** for high-bill accounts.
- **Cross-sell** to single-product users to deepen stickiness.
- **Proactive support** follow-ups after repeated tickets.

**Model performance (held-out test set):**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|--------:|-------:|----------:|-------:|---:|
| Logistic Regression | **0.826** | 0.592 | 0.43 | 0.73 | 0.54 |
| XGBoost | 0.806 | 0.537 | 0.59 | 0.33 | 0.42 |

> **Honest read:** the data's signal is largely linear, so the well-specified **logistic regression** edges out XGBoost on ranking (ROC-AUC). XGBoost is used for SHAP tree explanations and offers higher precision (fewer false alarms) at the default threshold. The threshold is a **business lever** — tune it to your retention budget (recall vs. precision).
>
> **Validation that the explanations are trustworthy:** the data is synthetic with *known* drivers, and SHAP recovered every top driver with the correct direction.

## Demo

![SHAP churn drivers](data/shap_drivers.png)

Per-customer reason codes (`explain_customer`):

```json
{
  "churn_probability": 0.78,
  "top_reasons": [
    {"feature": "contract_Month-to-month", "shap": 0.92, "effect": "+churn"},
    {"feature": "tenure_months",           "shap": 0.55, "effect": "+churn"},
    {"feature": "monthly_charges",         "shap": 0.31, "effect": "+churn"},
    {"feature": "premium_support",         "shap": -0.18, "effect": "-churn"}
  ]
}
```

## How it works

```
synthetic churn data ─► preprocess (one-hot + scale) ─► train ─┬─ LogisticRegression
                                                               └─ XGBoost
                                                                     │
                              evaluate (ROC-AUC, PR-AUC, P/R/F1)     │
                                                                     ▼
                                                   SHAP (TreeExplainer)
                                                          │
                          ┌───────────────────────────────┼───────────────────────────┐
                          ▼                                ▼                            ▼
                  global drivers + direction      per-customer reason codes     plain-English actions
```

## Tech stack

- **ML:** scikit-learn (LogReg, pipelines), XGBoost
- **Explainability:** SHAP (TreeExplainer)
- **Viz/App:** matplotlib, Plotly, Streamlit
- **Tests:** pytest (9 tests, including a check that SHAP recovers known drivers)

## Setup & run

```bash
cd 07-churn-prediction-shap
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data     # synthetic data/churn.csv
python -m src.run_pipeline      # train, evaluate, SHAP drivers, recommendations, plot
streamlit run dashboard.py      # interactive drivers + single-customer scoring
pytest -q
```

## Project structure

```
07-churn-prediction-shap/
├── dashboard.py            # Streamlit: drivers + per-customer reason codes
├── src/
│   ├── generate_data.py    # synthetic churn data with known drivers
│   ├── data.py             # load, split, preprocessing (ColumnTransformer)
│   ├── model.py            # LogReg + XGBoost, evaluation metrics
│   ├── explain.py          # SHAP drivers, recommendations, reason codes
│   └── run_pipeline.py     # end-to-end report + driver plot
├── notebooks/
│   └── churn_story.ipynb   # the analysis narrative
├── tests/                  # 9 pytest tests
├── requirements.txt
└── .gitignore
```

## Possible extensions

- **Uplift modeling** — target customers whose churn is *changeable*, not just likely.
- **Cost-sensitive thresholding** tied to customer lifetime value and offer cost.
- **Calibration** (Platt/isotonic) so probabilities are decision-ready.
- **Monitoring** for drift in drivers and score distribution over time.
- Export an **at-risk customer list** with reason codes to the CRM.
