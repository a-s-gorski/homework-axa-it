import pandas as pd
import pytest
from matplotlib.figure import Figure

from mlstream.pipelines.data_processing.nodes import preprocess_dataframe
from mlstream.pipelines.data_science.nodes import (
    split_train_test,
    train_or_search_model,
)
from mlstream.pipelines.reporting.nodes import (
    confusion_matrix_figure,
    metrics_json,
    predict_labels,
)


@pytest.mark.integration
def test_science_into_reporting(
    raw_training_df: pd.DataFrame,
    preprocess_params: dict,
    ds_split_params: dict,
    modeling_params: dict,
    reporting_params: dict,
):
    # Build X/y quickly
    df_proc = preprocess_dataframe(raw_training_df, preprocess_params)
    X_train, X_val, y_train, y_val = split_train_test(df_proc, ds_split_params)
    model, _ = train_or_search_model(X_train, y_train, X_val, y_val, modeling_params)

    # Reporting nodes
    y_pred = predict_labels(model, X_val)
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(X_val)

    mj = metrics_json(y_val, y_pred)
    for k in ["accuracy", "precision", "recall", "f1"]:
        assert k in mj
        assert isinstance(mj[k], float)

    fig = confusion_matrix_figure(
        y_true=y_val,
        y_pred=y_pred,
        labels=reporting_params["cm_labels"],
        normalize=reporting_params["cm_normalize"],
    )
    assert isinstance(fig, Figure)
