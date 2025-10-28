from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlstream.pipelines.data_processing.dataframe_processor import (
    DataFrameProcessor,
    DataFrameProcessorConfig,
    TargetStrategy,
)

logger = logging.getLogger(__name__)


def preprocess_dataframe(df: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    """
    Kedro node that preprocesses a DataFrame using DataFrameProcessor.
    Assumes **all** necessary options are passed via `parameters`.

    Expected `parameters` structure (keys are required unless marked optional):
      target:
        source: str                 # column to derive target from
        name: str                   # resulting target column name
        strategy: str               # "binary" | "multiclass" | "regression"
        # binary-only (optional):
        rule: str                   # "nonzero" | "threshold" | "mapping"
        threshold: float            # when rule == "threshold"
        mapping: dict               # when rule == "mapping"
      drop_columns: list[str]       # columns to drop
      categorical_columns: list[str]# columns to one-hot encode
      # optional:
      fillna: dict[str, Any]        # per-column NA fills
      invalid_target_replacement: int | float | str
      features: list[str]           # if provided, keep only these + target

    Returns:
      pd.DataFrame: processed DataFrame.
    """
    if not isinstance(parameters, dict):
        raise TypeError(
            "`parameters` must be a dict loaded by Kedro (e.g., params:preprocess)."
        )

    tgt = parameters["target"]
    strategy = TargetStrategy(tgt["strategy"])  # "binary" | "multiclass" | "regression"

    config = DataFrameProcessorConfig(
        columns_to_drop=list(parameters.get("drop_columns", [])),
        columns_to_onehotencode=list(parameters.get("categorical_columns", [])),
        columns_to_fillna=dict(parameters.get("fillna", {})),
        target_source=tgt["source"],
        target_name=tgt["name"],
        target_strategy=strategy,
        binary_rule=tgt.get("rule"),
        binary_threshold=tgt.get("threshold"),
        binary_mapping=tgt.get("mapping", {}),
        invalid_target_replacement=parameters.get("invalid_target_replacement"),
    )

    processor = DataFrameProcessor(config)
    df_proc = processor.process(df)

    features: list[str] | None = parameters.get("features")
    if features:
        keep = set(features) | {config.target_name}
        existing = [c for c in df_proc.columns if c in keep]
        if existing:
            if (
                config.target_name not in existing
                and config.target_name in df_proc.columns
            ):
                existing.append(config.target_name)
            df_proc = df_proc[existing]

    return df_proc
