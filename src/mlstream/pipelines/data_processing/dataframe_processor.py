"""
DataFrameProcessor
==================

Processor for pandas DataFrames driven by a Pydantic configuration
(`DataFrameProcessorConfig`). This module implements a deterministic
transformation pipeline:

Pipeline
--------
1) Fill NA values using `columns_to_fillna`.
2) Create/transform the target according to `target_strategy`
   (supports BINARY and REGRESSION).
3) One-hot encode columns listed in `columns_to_onehotencode`.
4) Drop columns listed in `columns_to_drop` (never drops the final target).

Key points
----------
- Runtime validation (`_validate`) performs light schema-ish checks and logs
  problems/warnings without mutating data.
- Binary targets support rules: {"nonzero", "threshold", "mapping"}.
- Regression targets coerce to numeric; invalids can be replaced with
  `invalid_target_replacement` if provided.
- One-hot encoding drops original categorical columns and joins dummy columns
  (with `drop_first=True` to avoid perfect multicollinearity).

Example
-------
    cfg = DataFrameProcessorConfig(
        columns_to_fillna={"age": 0},
        columns_to_onehotencode=["country"],
        columns_to_drop=["id"],
        target_source="score",
        target_name="label",
        target_strategy=TargetStrategy.BINARY,
        binary_rule="threshold",
        binary_threshold=0.5,
    )

    proc = DataFrameProcessor(cfg)
    df_out = proc.process(df_in)

Notes
-----
- Validation logs missing/extra columns and non-categorical one-hot columns
  (encoding still proceeds). Convert with `df[col] = df[col].astype("category")`
  if you want stricter dtype control.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .base_processor import BaseProcessor
from .dataframe_processor_config import DataFrameProcessorConfig, TargetStrategy

logger = logging.getLogger(__name__)


class DataFrameProcessor(BaseProcessor):
    """
    Processor for pandas DataFrames using a Pydantic-backed configuration.

    This class performs, in order:
      1) NA filling per `columns_to_fillna`
      2) Target creation/transformation per `target_*` fields
      3) One-hot encoding for columns in `columns_to_onehotencode`
      4) Dropping columns in `columns_to_drop` (never drops the final target)
    """

    def __init__(self, config: DataFrameProcessorConfig) -> None:
        """
        Initialize the processor.

        Args:
            config: Validated configuration for the processor.
        """
        self.config = config

    # ---------------------- Runtime validation helpers ----------------------

    def _validate(self, dataframe: pd.DataFrame | None = None) -> bool:
        """
        Runtime validation against a specific DataFrame (schema-ish checks).

        Args:
            dataframe: The DataFrame being processed.

        Returns:
            bool: True if validation completes (logs warnings/errors as needed).
        """
        if dataframe is None:
            raise ValueError("`dataframe` must be provided for validation.")

        self._validate_columns_present(dataframe)
        self._validate_find_extra_columns(dataframe)
        self._validate_columns_categorical(dataframe)
        return True

    def _validate_columns_present(self, dataframe: pd.DataFrame) -> None:
        """Validate that specified columns are present in the DataFrame."""
        required = (
            set(self.config.columns_to_drop)
            | set(self.config.columns_to_fillna.keys())
            | set(self.config.columns_to_onehotencode)
        )

        if self.config.target_source:
            required.add(self.config.target_source)

        missing = [c for c in required if c not in dataframe.columns]
        if missing:
            logger.error(f"Missing columns in DataFrame: {missing}")

    def _validate_find_extra_columns(self, dataframe: pd.DataFrame) -> None:
        """Warn if the DataFrame contains columns outside the configured set."""
        allowed = (
            set(self.config.columns_to_drop)
            | set(self.config.columns_to_fillna.keys())
            | set(self.config.columns_to_onehotencode)
        )

        if self.config.target_source:
            allowed.add(self.config.target_source)
        if self.config.target_name:
            allowed.add(self.config.target_name)

        extra = [c for c in dataframe.columns if c not in allowed]
        if extra:
            logger.warning(f"Extra columns in DataFrame: {extra}")

    def _validate_columns_categorical(self, dataframe: pd.DataFrame) -> None:
        """Warn if one-hot columns are not pandas 'category' dtype (still encodes)."""
        for col in self.config.columns_to_onehotencode:
            if col in dataframe.columns and not pd.api.types.is_categorical_dtype(
                dataframe[col]
            ):
                logger.warning(
                    f"Column '{col}' is not categorical (will still be one-hot encoded)."
                )

    @staticmethod
    def _drop_redundant_columns(
        dataframe: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame if present.

        Args:
            dataframe: Input DataFrame.
            columns: Columns to drop.

        Returns:
            DataFrame with columns removed.
        """
        cols = [c for c in columns if c in dataframe.columns]
        return dataframe.drop(columns=cols) if cols else dataframe

    @staticmethod
    def _one_hot_encode_columns(
        dataframe: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        """
        One-hot encode specified columns (drop originals, join dummies).

        Args:
            dataframe: Input DataFrame.
            columns: Categorical columns to encode.

        Returns:
            DataFrame with dummies joined.
        """
        if not columns:
            return dataframe
        df = dataframe.copy()
        cols_present = [c for c in columns if c in df.columns]
        if not cols_present:
            return df
        dummies = pd.get_dummies(
            df[cols_present], columns=cols_present, drop_first=True
        )
        df = df.drop(columns=cols_present)
        return df.join(dummies)

    @staticmethod
    def _fill_na(dataframe: pd.DataFrame, fill_map: dict[str, Any]) -> pd.DataFrame:
        """
        Fill NA values using a column->value map.

        Args:
            dataframe: Input DataFrame.
            fill_map: Mapping column -> fill value.

        Returns:
            DataFrame with NAs filled.
        """
        if not fill_map:
            return dataframe
        df = dataframe.copy()
        for col, val in fill_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _encode_binary_target(self, series: pd.Series) -> pd.Series:
        """
        Encode a Series into 0/1 according to `binary_rule`.

        Rules:
            - 'nonzero': 1 if value != 0 else 0
            - 'threshold': 1 if value > binary_threshold else 0
            - 'mapping': look up value in `binary_mapping` (unknowns use fallback)

        Args:
            series: Source series for target.

        Returns:
            Series of dtype int with 0/1 values.
        """
        rule = self.config.binary_rule or "nonzero"

        if rule == "nonzero":
            encoded = series.apply(lambda x: 1 if pd.notna(x) and x != 0 else 0)

        elif rule == "threshold":
            thr = float(self.config.binary_threshold or 0)
            encoded = series.apply(lambda x: 1 if pd.notna(x) and float(x) > thr else 0)

        elif rule == "mapping":
            mapping = self.config.binary_mapping
            encoded = series.map(mapping)
            if encoded.isna().any():
                if self.config.invalid_target_replacement is not None:
                    encoded = encoded.fillna(self.config.invalid_target_replacement)
                else:
                    encoded = encoded.fillna(0)
            encoded = encoded.astype(int)

        else:
            raise ValueError(f"Unsupported binary rule: {rule}")

        if self.config.invalid_target_replacement is not None:
            encoded = encoded.fillna(self.config.invalid_target_replacement)

        return encoded.astype(int)

    def _process_target_strategy(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create/transform the target column according to `target_strategy`.

        For BINARY:
            - Uses `_encode_binary_target`.

        For REGRESSION:
            - Coerces to numeric and fills invalids with `invalid_target_replacement` if provided.

        Args:
            dataframe: Input DataFrame.

        Returns:
            DataFrame with target column created/updated.
        """
        df = dataframe.copy()

        # If target already exists and no explicit source, assume it's final.
        if self.config.target_source is None and self.config.target_name in df.columns:
            return df

        source_col = self.config.target_source or self.config.target_name
        if source_col not in df.columns:
            raise KeyError(
                f"Target source column '{source_col}' not found in DataFrame."
            )

        src = df[source_col]

        if self.config.target_strategy == TargetStrategy.BINARY:
            df[self.config.target_name] = self._encode_binary_target(src)

        elif self.config.target_strategy == TargetStrategy.MULTICLASS:
            raise NotImplementedError("Multiclass target strategy not implemented yet.")

        elif self.config.target_strategy == TargetStrategy.REGRESSION:
            y = pd.to_numeric(src, errors="coerce")
            if self.config.invalid_target_replacement is not None:
                y = y.fillna(self.config.invalid_target_replacement)
            df[self.config.target_name] = y

        else:
            raise ValueError(
                f"Unsupported target strategy: {self.config.target_strategy}"
            )

        return df

    # ---------------------- Public API ----------------------

    def process(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame and return the processed result.

        Pipeline:
            1) Fill NA values
            2) Create/transform target
            3) One-hot encode categorical columns
            4) Drop redundant columns (never drops the final target)

        Args:
            dataframe: The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self._validate(dataframe)

        df = dataframe.copy()
        df = self._fill_na(df, self.config.columns_to_fillna)
        df = self._process_target_strategy(df)
        df = self._one_hot_encode_columns(df, self.config.columns_to_onehotencode)

        # protect the target from being dropped accidentally
        safe_drop = [
            c for c in self.config.columns_to_drop if c != self.config.target_name
        ]
        df = self._drop_redundant_columns(df, safe_drop)

        return df
