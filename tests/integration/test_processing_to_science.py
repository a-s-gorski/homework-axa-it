import pandas as pd
import pytest

from mlstream.pipelines.data_processing.nodes import preprocess_dataframe
from mlstream.pipelines.data_science.nodes import (
    split_train_test,
    train_or_search_model,
)


@pytest.mark.integration
def test_processing_into_science_training(
    raw_training_df: pd.DataFrame,
    preprocess_params: dict,
    ds_split_params: dict,
    modeling_params: dict,
):
    # 1) process
    df_proc = preprocess_dataframe(raw_training_df, preprocess_params)
    assert "target" in df_proc.columns
    assert "junk_col" not in df_proc.columns
    # one-hot presence (at least one col created)
    assert any(c.startswith("feat_cat_") for c in df_proc.columns)

    # 2) split
    X_train, X_val, y_train, y_val = split_train_test(df_proc, ds_split_params)
    assert len(X_train) + len(X_val) == len(df_proc)
    assert set(X_train.columns) == set(X_val.columns)
    assert y_train.name == "target"
    assert y_val.name == "target"

    # 3) train
    model, info = train_or_search_model(X_train, y_train, X_val, y_val, modeling_params)
    assert hasattr(model, "predict")
    assert "best_params" in info
    assert info["mode"] in {"direct", "optuna"}

    # basic sanity: model predicts 0/1 and aligns to X_val index
    y_pred = model.predict(X_val)
    assert set(pd.Series(y_pred).unique()).issubset({0, 1})
    assert len(y_pred) == len(X_val)
