# tests/pipelines/data_processing/test_preprocess_dataframe.py
import pandas as pd
import pytest

from mlstream.pipelines.data_processing.nodes import preprocess_dataframe


def test_parameters_type_guard():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="`parameters` must be a dict"):
        preprocess_dataframe(df, parameters=["not-a-dict"])  # type: ignore[arg-type]


def test_missing_target_source_raises_keyerror():
    df = pd.DataFrame({"other": [1, 2]})
    params = {
        "target": {
            "source": "missing",
            "name": "y",
            "strategy": "binary",
            "rule": "nonzero",
        },
        "drop_columns": [],
        "categorical_columns": [],
    }
    with pytest.raises(KeyError, match="Target source column 'missing' not found"):
        preprocess_dataframe(df, params)


def test_binary_nonzero_rule():
    df = pd.DataFrame({"x": [0, 1, -2, None]})
    params = {
        "target": {"source": "x", "name": "y", "strategy": "binary", "rule": "nonzero"},
        "drop_columns": [],
        "categorical_columns": [],
    }
    out = preprocess_dataframe(df, params)
    assert out["y"].tolist() == [0, 1, 1, 0]


def test_binary_threshold_rule():
    df = pd.DataFrame({"score": [0, 0.5, 1.1, "2.0", None]})
    params = {
        "target": {
            "source": "score",
            "name": "y",
            "strategy": "binary",
            "rule": "threshold",
            "threshold": 1.0,
        },
        "drop_columns": [],
        "categorical_columns": [],
    }
    out = preprocess_dataframe(df, params)
    assert out["y"].tolist() == [0, 0, 1, 1, 0]


def test_binary_mapping_with_invalid_replacement():
    df = pd.DataFrame({"status": ["OK", "WARN", "UNKNOWN", None]})
    params = {
        "target": {
            "source": "status",
            "name": "y",
            "strategy": "binary",
            "rule": "mapping",
            "mapping": {"OK": 0, "WARN": 1, "FAIL": 1},
        },
        "drop_columns": [],
        "categorical_columns": [],
        "invalid_target_replacement": 0,
    }
    out = preprocess_dataframe(df, params)
    assert out["y"].tolist() == [0, 1, 0, 0]


def test_regression_with_invalid_replacement():
    df = pd.DataFrame({"val": ["1.2", "bad", 3.0, None]})
    params = {
        "target": {"source": "val", "name": "y", "strategy": "regression"},
        "drop_columns": [],
        "categorical_columns": [],
        "invalid_target_replacement": -1.0,
    }
    out = preprocess_dataframe(df, params)
    assert out["y"].tolist() == [1.2, -1.0, 3.0, -1.0]


def test_fillna_applied_before_target_creation():
    df = pd.DataFrame({"score": [None, 2, None]})
    params = {
        "target": {
            "source": "score",
            "name": "y",
            "strategy": "binary",
            "rule": "threshold",
            "threshold": 3,
        },
        "drop_columns": [],
        "categorical_columns": [],
        "fillna": {"score": 5},  # should fill before evaluating threshold
    }
    out = preprocess_dataframe(df, params)
    assert out["y"].tolist() == [1, 0, 1]


def test_one_hot_encoding_and_safe_drop():
    df = pd.DataFrame(
        {
            "cat": [
                "A",
                "B",
                "C",
                "A",
            ],  # not categorical dtype on purpose; still encodes
            "x": [1, 2, 3, 4],
        }
    )
    params = {
        "target": {"source": "x", "name": "y", "strategy": "regression"},
        "drop_columns": ["x", "y"],  # target must be protected from drop
        "categorical_columns": ["cat"],  # will be one-hot encoded with drop_first=True
    }
    out = preprocess_dataframe(df, params)

    # original 'cat' should be gone; dummies present (drop_first=True -> two columns)
    assert "cat" not in out.columns
    assert set(col for col in out.columns if col.startswith("cat_")) == {
        "cat_B",
        "cat_C",
    }

    # source 'x' dropped; target 'y' preserved
    assert "x" not in out.columns
    assert "y" in out.columns
    assert out["y"].tolist() == [1, 2, 3, 4]


# ------------------------------ Features filter ------------------------------


def test_features_filter_keeps_only_requested_plus_target():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "cat": ["A", "A", "B"],
        }
    )
    params = {
        "target": {"source": "a", "name": "y", "strategy": "regression"},
        "drop_columns": [],
        "categorical_columns": ["cat"],  # will add dummy(s)
        "features": ["b"],  # keep only 'b' + 'y'
    }
    out = preprocess_dataframe(df, params)

    # Only 'b' and target 'y' should remain
    assert set(out.columns) == {"b", "y"}
    assert out["y"].tolist() == [1, 2, 3]
    assert out["b"].tolist() == [10, 20, 30]
