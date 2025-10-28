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
