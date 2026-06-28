from src.data import split
from src.generate_data import generate_dataset
from src.model import build_logreg, build_xgboost, evaluate, train_and_evaluate


def _ds(n=2000):
    return split(generate_dataset(n=n, seed=7))


def test_logreg_learns_signal():
    ds = _ds()
    pipe = build_logreg().fit(ds.X_train, ds.y_train)
    res = evaluate("lr", pipe, ds)
    assert res.roc_auc > 0.7  # well above chance
    assert 0 <= res.precision <= 1 and 0 <= res.recall <= 1


def test_xgboost_learns_signal():
    ds = _ds()
    pipe = build_xgboost().fit(ds.X_train, ds.y_train)
    res = evaluate("xgb", pipe, ds)
    assert res.roc_auc > 0.7


def test_train_and_evaluate_returns_both_models():
    ds = _ds(1500)
    out = train_and_evaluate(ds)
    assert set(out["fitted"]) == {"LogisticRegression", "XGBoost"}
    assert len(out["results"]) == 2
