from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .models import retrieve_model_class as _get_model_class
from .optuna_train import handle_optuna_search


def split_train_test(
    df: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Kedro node: split features/target into train/test using config.

    Inputs:
      - df: The processed dataset (must contain the target column).
      - parameters (from params:data_science), expected keys:
          target_column: str                      # e.g., "target"
          test_size: float                        # e.g., 0.2
          random_state: int                       # e.g., 42
          stratify: bool (optional, default True) # if True, stratify by target

    Outputs:
      - X_train, X_test, y_train, y_test
    """
    target_col: str = parameters["target_column"]
    test_size: float = float(parameters["test_size"])
    random_state: int = int(parameters["random_state"])
    stratify_enabled: bool = bool(parameters.get("stratify", True))

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if stratify_enabled else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_val, y_train, y_val


def train_or_search_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters_modeling: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Kedro node: either train a model directly or run Optuna hyperparameter search.

    Inputs:
      - X_train, y_train
      - parameters_modeling (from params:modeling), expected structure:
          model:
            type: lightgbm | random_forest | logreg | xgboost
            params: {...}               # base params for direct training or as defaults in search
          search:
            enabled: bool
            n_trials: int
            direction: maximize | minimize
            test_size: float            # validation split size (e.g., 0.2)
            space:                      # per-model search space (see examples below)
              n_estimators: {type: int, low: 10, high: 200}
              learning_rate: {type: float, low: 0.001, high: 0.3, log: true}
              ...
          metric: accuracy              # (currently supports 'accuracy')

      - parameters_ds (from params:data_science), expected keys:
          random_state: int             # reused for the inner validation split during search

    Outputs:
      - fitted model
      - info dict (e.g., {"best_params": {...}, "score": float})
    """
    model_cfg = parameters_modeling["model"]
    model_name: str = model_cfg["type"]
    base_params: dict[str, Any] = dict(model_cfg.get("params", {}))

    metric_name: str = parameters_modeling.get("metric", "accuracy")
    search_cfg: dict[str, Any] = parameters_modeling.get("search", {"enabled": False})

    model_class = _get_model_class(model_name)

    if not search_cfg.get("enabled", False):
        model = model_class(**base_params)
        model.fit(X_train, y_train)
        return model, {"best_params": base_params, "score": None, "mode": "direct"}

    best_params, best_value, n_trials = handle_optuna_search(
        search_cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        base_params,
        model_class,
        model_name,
        metric_name,
    )

    model = model_class(**best_params)
    model.fit(X_train, y_train)

    return model, {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "mode": "optuna",
    }
