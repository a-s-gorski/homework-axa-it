import pandas as pd
import pytest

import mlstream.pipelines.data_science.optuna_train as optuna_mod
from mlstream.pipelines.data_science.optuna_train import handle_optuna_search


class FakeTrial:
    """Simple trial that just carries a number."""
    def __init__(self, number: int):
        self.number = number


class FakeStudy:
    """Drop-in for optuna Study that calls the objective N times and records the best."""
    def __init__(self, direction: str = "maximize"):
        assert direction in ("maximize", "minimize")
        self.direction = direction
        self.best_value = None
        self.best_params = {}
        self.trials = []
        self._comp = (lambda a, b: a > b) if direction == "maximize" else (lambda a, b: a < b)

    def optimize(self, objective, n_trials: int, show_progress_bar: bool = False):
        for i in range(int(n_trials)):
            trial = FakeTrial(i)
            val = objective(trial)  # may raise (we let it bubble up)
            self.trials.append({"number": i, "value": val})
            if self.best_value is None or self._comp(val, self.best_value):
                self.best_value = val
                # objective uses _suggest_from_space(trial, space)
                # our stub will return a dict depending on trial.number,
                # and handle_optuna_search will merge it as study.best_params
                self.best_params = dict(_LAST_SUGGESTED_PARAMS)  # captured via stub


# Model that records init kwargs and produces deterministic predictions
class RecorderModel:
    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        # Base rule: class = 1 if first feature > 0 else 0
        preds = (X.iloc[:, 0] > 0).astype(int).values
        flip = int(self.init_kwargs.get("flip", 0))
        if flip == 1:
            preds = 1 - preds
        return preds


# global used to capture suggestions from our stub
_LAST_SUGGESTED_PARAMS = {}



@pytest.fixture
def toy_split():
    # Simple 1-D feature; y = (x > 0)
    X_train = pd.DataFrame({"f0": [-2.0, -1.0, 1.0, 2.0]})
    y_train = pd.Series([0, 0, 1, 1], name="y")
    X_val = pd.DataFrame({"f0": [-0.5, 0.5]})
    y_val = pd.Series([0, 1], name="y")
    return X_train, y_train, X_val, y_val



def test_handle_optuna_search_returns_best_params_and_value(monkeypatch, toy_split):
    # Stub optuna.create_study -> FakeStudy
    monkeypatch.setattr(optuna_mod.optuna, "create_study", lambda direction="maximize": FakeStudy(direction))

    # Stub _suggest_from_space so trial 0 is bad (flip=1), trial 1 is good (flip=0)
    def stub_suggest(trial, space):
        global _LAST_SUGGESTED_PARAMS
        _LAST_SUGGESTED_PARAMS = {"flip": 1 if trial.number == 0 else 0}
        return dict(_LAST_SUGGESTED_PARAMS)

    monkeypatch.setattr(optuna_mod, "_suggest_from_space", stub_suggest)

    # Model resolver is passed in as model_class already (so we skip _get_model_class)
    model_class = RecorderModel

    X_train, y_train, X_val, y_val = toy_split

    search_cfg = {
        "enabled": True,
        "n_trials": 2,
        "direction": "maximize",
        "space": {"flip": {"type": "int", "low": 0, "high": 1}},
    }
    base_params = {"alpha": 5}
    best_params, best_value, n_trials = handle_optuna_search(
        search_cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        base_params,
        model_class,
        model_name="xgboost",
        metric_name="accuracy",
    )

    # Trial 1 should win (flip=0 yields perfect accuracy on our toy val set)
    assert best_params == {"alpha": 5, "flip": 0}
    assert best_value == 1.0
    assert n_trials == 2


def test_unsupported_metric_raises(monkeypatch, toy_split):
    # Ensure objective raises and bubbles up when metric is unsupported
    monkeypatch.setattr(optuna_mod.optuna, "create_study", lambda direction="maximize": FakeStudy(direction))

    def stub_suggest(trial, space):
        global _LAST_SUGGESTED_PARAMS
        _LAST_SUGGESTED_PARAMS = {}
        return {}

    monkeypatch.setattr(optuna_mod, "_suggest_from_space", stub_suggest)

    X_train, y_train, X_val, y_val = toy_split

    search_cfg = {
        "enabled": True,
        "n_trials": 1,
        "direction": "maximize",
        "space": {},
    }
    with pytest.raises(ValueError, match="Unsupported metric"):
        handle_optuna_search(
            search_cfg,
            X_train,
            y_train,
            X_val,
            y_val,
            base_params={},
            model_class=RecorderModel,
            model_name="xgboost",
            metric_name="rmse",  # not supported
        )
