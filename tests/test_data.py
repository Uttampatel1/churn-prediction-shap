from src.data import CATEGORICAL, NUMERIC, build_preprocessor, feature_names, split
from src.generate_data import generate_dataset


def test_generate_dataset_shape_and_label():
    df = generate_dataset(n=500, seed=1)
    assert len(df) == 500
    assert set(df["churn"].unique()) <= {0, 1}
    assert 0.05 < df["churn"].mean() < 0.6


def test_split_stratified_shapes():
    df = generate_dataset(n=1000, seed=2)
    ds = split(df, test_size=0.25)
    assert len(ds.X_train) == 750 and len(ds.X_test) == 250
    assert list(ds.X_train.columns) == NUMERIC + CATEGORICAL


def test_feature_names_after_onehot():
    df = generate_dataset(n=400, seed=3)
    ds = split(df)
    prep = build_preprocessor().fit(ds.X_train)
    names = feature_names(prep)
    # numerics + expanded one-hot categoricals
    assert names[: len(NUMERIC)] == NUMERIC
    assert any("contract_" in n for n in names)
