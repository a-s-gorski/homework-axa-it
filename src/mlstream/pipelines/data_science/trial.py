from typing import Any

from optuna.trial import Trial


def extract_suggested_params(
    trial: Trial, space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Convert a YAML-defined parameter space to Optuna suggestions.

    space example (per-model):
      n_estimators: {type: int, low: 10, high: 200}
      learning_rate: {type: float, low: 0.001, high: 0.3, log: true}
      max_depth: {type: int, low: 3, high: 10}
      subsample: {type: float, low: 0.5, high: 1.0}

    Returns a dict of suggested hyperparameters for this trial.
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
