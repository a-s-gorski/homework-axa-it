"""
Module: trial
-------------
Provides utilities for parsing and applying YAML-defined hyperparameter search spaces
within Optuna optimization trials.

This module defines a helper function `extract_suggested_params` that converts a
dictionary-based parameter space (often defined in YAML or JSON) into actual parameter
suggestions for a given Optuna trial.

Functions
---------
extract_suggested_params(trial, space)
    Converts a parameter search space into Optuna trial suggestions for supported types:
    int, float, and categorical.

Usage example
-------------
>>> from optuna import Trial
>>> from mypkg.trial import extract_suggested_params
>>> space = {
...     "n_estimators": {"type": "int", "low": 10, "high": 200},
...     "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
...     "max_depth": {"type": "int", "low": 3, "high": 10},
...     "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
... }
>>> def objective(trial: Trial):
...     params = extract_suggested_params(trial, space)
...     print(params)
...     return 0.0
"""

from typing import Any

from optuna.trial import Trial


def extract_suggested_params(
    trial: Trial, space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Convert a YAML-defined parameter space to Optuna suggestions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current Optuna trial object, used to suggest parameter values.
    space : dict[str, dict[str, Any]]
        The search space definition. Each entry should define:
            - "type": one of {"int", "float", "categorical"}
            - "low" and "high" for numeric ranges (int or float)
            - optional "log": bool for logarithmic sampling (floats)
            - "choices" for categorical parameters

        Example:
        {
            "n_estimators": {"type": "int", "low": 10, "high": 200},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
            "booster": {"type": "categorical", "choices": ["gbtree", "dart"]}
        }

    Returns
    -------
    dict[str, Any]
        Dictionary of suggested hyperparameter values for this trial.

    Raises
    ------
    ValueError
        If a parameter type is unsupported or improperly specified.

    Examples
    --------
    >>> def objective(trial):
    ...     space = {
    ...         "depth": {"type": "int", "low": 3, "high": 10},
    ...         "lr": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True}
    ...     }
    ...     return extract_suggested_params(trial, space)
    """
    suggested: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            suggested[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"])
            )
        elif t == "float":
            suggested[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif t == "categorical":
            suggested[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported search space type for '{name}': {t}")
    return suggested
