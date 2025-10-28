"""
Module: optuna_search
---------------------
Provides functionality for performing hyperparameter optimization using Optuna.

This module defines a helper function `handle_optuna_search` that runs an Optuna study
on a given model class, using a defined search space and evaluation metric. It integrates
base model parameters, handles LightGBM-specific defaults, and returns the best parameters
and corresponding metric score.

Dependencies
------------
- optuna (required for hyperparameter tuning)
- scikit-learn (for accuracy metrics)

Functions
---------
handle_optuna_search(search_cfg, X_train, y_train, X_val, y_val, base_params, model_class, model_name, metric_name)
    Executes Optuna-based hyperparameter search and returns the best parameters,
    the best metric value, and the number of trials performed.

Usage example
-------------
>>> from lightgbm import LGBMClassifier
>>> from mypkg.optuna_search import handle_optuna_search
>>> search_cfg = {
...     "space": {
...         "n_estimators": {"type": "int", "low": 50, "high": 300},
...         "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
...     },
...     "n_trials": 20,
...     "direction": "maximize"
... }
>>> X_train, X_val, y_train, y_val = load_data()
>>> base_params = {"random_state": 42}
>>> best_params, best_score, n_trials = handle_optuna_search(
...     search_cfg, X_train, y_train, X_val, y_val,
...     base_params, LGBMClassifier, "lightgbm", "accuracy"
... )
>>> print(best_params, best_score)
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score

from .trial import extract_suggested_params as _suggest_from_space

try:
    import optuna
except ImportError as e:
    raise RuntimeError(
        "Optuna is required for hyperparameter search. Install `optuna`."
    ) from e


def handle_optuna_search(
    search_cfg: dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    base_params: dict[str, Any],
    model_class: type,
    model_name: str,
    metric_name: str,
) -> tuple[dict[str, Any], float, int]:  # noqa: PLR0913
    """
    Run Optuna hyperparameter search for a given model and dataset.

    Parameters
    ----------
    search_cfg : dict[str, Any]
        Configuration dictionary for the search, containing:
        - "space": a dictionary defining parameter search space for Optuna.
        - "n_trials": number of trials to run.
        - "direction": optimization direction, either "maximize" or "minimize".
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    X_val : array-like
        Validation feature matrix.
    y_val : array-like
        Validation target vector.
    base_params : dict[str, Any]
        Base model parameters to include in every trial.
    model_class : type
        The model class to optimize, e.g., `LGBMClassifier` or `RandomForestClassifier`.
    model_name : str
        A string identifier for the model, used to apply model-specific defaults.
        Supported: "lightgbm", "random_forest", "logreg".
    metric_name : str
        Name of the evaluation metric. Currently supports only "accuracy".

    Returns
    -------
    tuple[dict[str, Any], float, int]
        A tuple containing:
        - best_params : dict[str, Any]
            Combined base and best-found hyperparameters.
        - best_value : float
            Best score achieved according to the specified metric.
        - n_trials : int
            Total number of trials executed during optimization.

    Raises
    ------
    RuntimeError
        If Optuna is not installed.
    ValueError
        If the specified metric or model name is unsupported.

    Notes
    -----
    - LightGBM models are automatically configured with `objective="binary"`,
      `n_jobs=-1`, and `verbosity=-1` if not specified.
    - Extend the metric handling section to support additional metrics as needed.
    """
    space: dict[str, dict[str, Any]] = dict(search_cfg["space"])
    direction: str = search_cfg.get("direction", "maximize")

    def objective(trial) -> float:
        suggested = _suggest_from_space(trial, space)
        params = {**base_params, **suggested}

        if model_name == "lightgbm":
            params.setdefault("objective", "binary")
            params.setdefault("n_jobs", -1)
            params.setdefault("verbosity", -1)

        model = model_class(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric_name == "accuracy":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError(f"Unsupported metric '{metric_name}'. Add it if needed.")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(
        objective, n_trials=int(search_cfg["n_trials"]), show_progress_bar=False
    )

    best_params = {**base_params, **study.best_params}
    return best_params, study.best_value, len(study.trials)
