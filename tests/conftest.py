import numpy as np
import pandas as pd
import pytest


# ---------- Synthetic toy data ----------
@pytest.fixture(scope="session")
def raw_training_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "feat_num1": rng.normal(0, 1, n),
            "feat_num2": rng.normal(3, 2, n),
            "feat_cat": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
            "target_src": rng.integers(0, 3, n),
            "junk_col": 1,  # will be dropped
        }
    )
    # Make a roughly linearly separable binary target from target_src for easier testing
    df["target_src"] = (df["target_src"] > 0).astype(int)
    return df


# ---------- Parameters for data_processing.preprocess_dataframe ----------
@pytest.fixture(scope="session")
def preprocess_params() -> dict:
    return {
        "target": {
            "source": "target_src",
            "name": "target",
            "strategy": "binary",  # maps to TargetStrategy
            "rule": "mapping",
            "mapping": {0: 0, 1: 1},  # explicit, though unnecessary since already 0/1
        },
        "drop_columns": ["junk_col"],
        "categorical_columns": ["feat_cat"],
        "fillna": {},
        "invalid_target_replacement": 0,
        # keep features if you want to prove column selection works:
        # "features": ["feat_num1", "feat_num2", "feat_cat_A", "feat_cat_B", "feat_cat_C"],
    }


# ---------- Parameters for data_science.split_train_test ----------
@pytest.fixture(scope="session")
def ds_split_params() -> dict:
    return {
        "target_column": "target",
        "test_size": 0.25,
        "random_state": 123,
        "stratify": True,
    }


# ---------- Parameters for data_science.train_or_search_model ----------
@pytest.fixture(scope="session")
def modeling_params() -> dict:
    # keep search disabled for deterministic tests; switch on when you want to exercise Optuna
    return {
        "model": {"type": "logreg", "params": {"max_iter": 200}},
        "search": {"enabled": False},
        "metric": "accuracy",
    }


# ---------- Parameters for reporting pipeline ----------
@pytest.fixture(scope="session")
def reporting_params() -> dict:
    return {
        "cm_labels": [0, 1],
        "cm_normalize": None,  # can be None|"true"|"pred"|"all"
    }
