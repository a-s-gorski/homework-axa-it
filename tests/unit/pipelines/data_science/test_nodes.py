import numpy as np
import pandas as pd
import pytest

from mlstream.pipelines.data_science.nodes import (
    split_train_test,
    train_or_search_model,
)


def _make_df(n=100, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)  # binary for stratify
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    df = pd.DataFrame(cols)
    df["target"] = y
    return df


def test_split_train_test_basic_stratified():
    df = _make_df(n=50, n_features=2, seed=1)
    params = {
        "target_column": "target",
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    }
    X_tr, X_te, y_tr, y_te = split_train_test(df, params)

    # sizes
    assert len(X_tr) + len(X_te) == len(df)
    assert len(y_tr) + len(y_te) == len(df)
    assert X_tr.shape[1] == df.shape[1] - 1  # all features except target

    # target column not present in features
    assert "target" not in X_tr.columns
    assert "target" not in X_te.columns

    # stratification should approximately preserve class distribution
    p_all = df["target"].mean()
    p_tr = y_tr.mean()
    p_te = y_te.mean()
    assert abs(p_tr - p_all) < 0.1
    assert abs(p_te - p_all) < 0.1


def test_split_train_test_no_stratify():
    df = _make_df(n=40, n_features=2, seed=2)
    params = {
        "target_column": "target",
        "test_size": 0.25,
        "random_state": 0,
        "stratify": False,
    }
    X_tr, X_te, y_tr, y_te = split_train_test(df, params)
    assert len(X_te) == int(0.25 * len(df))


def test_split_train_test_missing_target_raises():
    df = _make_df().drop(columns=["target"])
    params = {"target_column": "target", "test_size": 0.2, "random_state": 0}
    with pytest.raises(KeyError, match="Target column 'target' not found"):
        split_train_test(df, params)


# --------------------------- train_or_search_model ----------------------------


class _RecorderModel:
    """A tiny stand-in for real estimators that records init kwargs & fit calls."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fitted = False
        self.fit_X_shape = None
        self.fit_y_len = None

    def fit(self, X, y):
        self.fitted = True
        self.fit_X_shape = X.shape
        self.fit_y_len = len(y)
        return self


def test_train_or_search_model_direct_path(monkeypatch):
    # monkeypatch the model resolver to return our recorder class
    import mlstream.pipelines.data_science.nodes as mod

    def fake_get_model_class(name):
        assert name == "random_forest"  # from params below
        return _RecorderModel

    monkeypatch.setattr(mod, "_get_model_class", fake_get_model_class)

    # create simple split
    df = _make_df(n=30, n_features=2, seed=3)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_tr, X_te = X.iloc[:20], X.iloc[20:]
    y_tr, y_te = y.iloc[:20], y.iloc[20:]

    params_modeling = {
        "model": {
            "type": "random_forest",
            "params": {"n_estimators": 123, "max_depth": 7},
        },
        # search disabled
        "search": {"enabled": False},
        "metric": "accuracy",  # default anyway
    }

    model, info = train_or_search_model(X_tr, y_tr, X_te, y_te, params_modeling)

    assert isinstance(model, _RecorderModel)
    assert model.fitted is True
    assert model.fit_X_shape == X_tr.shape
    assert model.fit_y_len == len(y_tr)
    assert model.init_kwargs == {"n_estimators": 123, "max_depth": 7}

    # info dict for direct mode
    assert info["best_params"] == {"n_estimators": 123, "max_depth": 7}
    assert info["score"] is None
    assert info["mode"] == "direct"


def test_train_or_search_model_optuna_path(monkeypatch):
    import mlstream.pipelines.data_science.nodes as mod

    # capture that handle_optuna_search is called with expected arguments
    captured = {}

    def fake_get_model_class(name):
        assert name == "xgboost"
        return _RecorderModel

    def fake_handle_optuna_search(
        search_cfg,
        X_tr,
        y_tr,
        X_val,
        y_val,
        base_params,
        model_class,
        model_name,
        metric_name,
    ):
        # record call args for assertions
        captured.update(
            dict(
                search_cfg=search_cfg,
                X_tr_shape=X_tr.shape,
                y_tr_len=len(y_tr),
                X_val_shape=X_val.shape,
                y_val_len=len(y_val),
                base_params=base_params,
                model_class=model_class,
                model_name=model_name,
                metric_name=metric_name,
            )
        )
        # Return (best_params, best_value, n_trials)
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}, 0.789, 25

    monkeypatch.setattr(mod, "_get_model_class", fake_get_model_class)
    monkeypatch.setattr(mod, "handle_optuna_search", fake_handle_optuna_search)

    # fake split
    df = _make_df(n=60, n_features=3, seed=4)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_tr, X_te = X.iloc[:45], X.iloc[45:]
    y_tr, y_te = y.iloc[:45], y.iloc[45:]

    params_modeling = {
        "model": {
            "type": "xgboost",
            "params": {"n_estimators": 50, "max_depth": 3},  # base defaults for search
        },
        "search": {
            "enabled": True,
            "n_trials": 25,
            "direction": "maximize",
            "test_size": 0.2,
            "space": {"n_estimators": {"type": "int", "low": 10, "high": 300}},
        },
        "metric": "accuracy",  # ensure it’s forwarded
    }

    model, info = train_or_search_model(X_tr, y_tr, X_te, y_te, params_modeling)

    # model is trained with returned best_params
    assert isinstance(model, _RecorderModel)
    assert model.fitted is True
    assert model.init_kwargs == {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
    }

    # check the search call received expected inputs
    assert captured["search_cfg"]["enabled"] is True
    assert captured["X_tr_shape"] == X_tr.shape
    assert captured["y_tr_len"] == len(y_tr)
    assert captured["X_val_shape"] == X_te.shape
    assert captured["y_val_len"] == len(y_te)
    assert captured["base_params"] == {"n_estimators": 50, "max_depth": 3}
    assert captured["model_class"] is _RecorderModel
    assert captured["model_name"] == "xgboost"
    assert captured["metric_name"] == "accuracy"

    # info dict for optuna mode
    assert info["best_params"] == {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
    }
    assert info["best_value"] == 0.789
    assert info["n_trials"] == 25
    assert info["mode"] == "optuna"
