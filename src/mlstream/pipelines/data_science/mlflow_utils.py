import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def mlflow_autolog_run(params_modeling_index: int = -1):
    """
    Decorator that enables MLflow autologging if parameters_modeling["tracking"]["mlflow_enabled"] is True.
    Modularized into helper functions and includes logging for MLflow enablement status.
    Works for functions returning (model, info_dict).
    """

    # ---------- Internal helpers ----------
    def _is_mlflow_enabled(*args: list) -> tuple[bool, dict]:
        params_modeling = args[0][
            params_modeling_index
        ]  # need to first convert tuple to list, then extract last dict
        tracking_cfg = (params_modeling or {}).get("tracking", {})
        enabled = tracking_cfg.get("mlflow_enabled", False)
        return enabled, tracking_cfg

    def _setup_mlflow(tracking_cfg: dict):
        import mlflow  # noqa: PLC0415

        uri = tracking_cfg.get("tracking_uri")
        experiment = tracking_cfg.get("experiment_name", "default")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.sklearn.autolog()
        logger.info(
            f"MLflow tracking initialized: experiment='{experiment}', uri='{uri or 'default local'}'"
        )
        return mlflow

    def _log_model_params(mlflow, model_cfg: dict):
        mlflow.log_param("model_type", model_cfg.get("type", "unknown"))
        for k, v in model_cfg.get("params", {}).items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(k, v)

    def _log_info_dict(mlflow, info: dict):
        if not isinstance(info, dict):
            return
        for k, v in info.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float, str, bool)):
                        mlflow.log_param(f"{k}.{kk}", vv)

    def _log_model_artifact(mlflow, model):
        try:
            import mlflow.sklearn  # noqa: PLC0415

            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            logger.warning(f"[mlflow_autolog_run] Model logging skipped: {e}")

    # ---------- Main decorator ----------
    def decorator(func: Callable[..., tuple[Any, dict[str, Any]]]):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            enabled, tracking_cfg = _is_mlflow_enabled(args)

            if not enabled:
                logger.info(
                    f"MLflow tracking disabled for '{func.__name__}'. Executing function without logging."
                )
                return func(*args, **kwargs)

            logger.info(
                f"MLflow tracking enabled for '{func.__name__}'. Starting run..."
            )

            # Lazy import mlflow only if enabled
            mlflow = _setup_mlflow(tracking_cfg)

            params_modeling = args[params_modeling_index]
            model_cfg = params_modeling.get("model", {})

            with mlflow.start_run(run_name=func.__name__):
                logger.debug(f"Started MLflow run for function: {func.__name__}")
                _log_model_params(mlflow, model_cfg)

                model, info = func(*args, **kwargs)

                _log_info_dict(mlflow, info)
                _log_model_artifact(mlflow, model)

                logger.info(
                    f"Completed MLflow run for '{func.__name__}' and logged model successfully."
                )
                return model, info

        return wrapper

    return decorator
