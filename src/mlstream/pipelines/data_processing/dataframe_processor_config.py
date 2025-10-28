"""
DataFrameProcessor Configurations
=================================

This module defines configuration models for the MLStream DataFrame processing
pipeline, using Pydantic v2 for validation and normalization.

Contents
--------
- `TargetStrategy`: Enum of supported target-generation strategies.
- `DataFrameProcessorConfig`: Serializable config capturing column transforms
  (drop, one-hot encode, fillna) and target derivation rules.

Key behaviors
-------------
- `binary_rule` is normalized to one of: {"nonzero", "threshold", "mapping"}.
- Cross-field checks enforce required params for BINARY targets:
    * `threshold` rule -> `binary_threshold` must be set.
    * `mapping` rule   -> `binary_mapping` must be non-empty.

Example
-------
Basic nonzero rule:
    >>> DataFrameProcessorConfig(
    ...     columns_to_drop=["id"],
    ...     columns_to_onehotencode=["country"],
    ...     columns_to_fillna={"age": 0},
    ...     target_source="label_raw",
    ...     target_name="label",
    ...     target_strategy=TargetStrategy.BINARY,
    ...     binary_rule="nonzero",
    ... )

Threshold rule:
    >>> DataFrameProcessorConfig(
    ...     target_source="score",
    ...     target_strategy=TargetStrategy.BINARY,
    ...     binary_rule="threshold",
    ...     binary_threshold=0.5,
    ... )

Mapping rule with fallback:
    >>> DataFrameProcessorConfig(
    ...     target_source="status",
    ...     target_strategy=TargetStrategy.BINARY,
    ...     binary_rule="mapping",
    ...     binary_mapping={"OK": 0, "WARN": 1, "FAIL": 1},
    ...     invalid_target_replacement=0,
    ... )

Serialization (e.g., to save with a run config):
    >>> cfg = DataFrameProcessorConfig()
    >>> cfg.model_dump()   # dict ready for YAML/JSON
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TargetStrategy(str, Enum):
    """Supported strategies for generating the target column."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class DataFrameProcessorConfig(BaseModel):
    """
    Pydantic configuration for `DataFrameProcessor`.

    This model captures static, serializable configuration for transforming
    pandas DataFrames, including target creation, one-hot encoding, NA handling,
    and column dropping.
    """

    # Generic transformations
    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Columns to drop at the end of processing (target-safe).",
    )
    columns_to_onehotencode: list[str] = Field(
        default_factory=list,
        description="Categorical columns to one-hot encode (will be dropped after encoding).",
    )
    columns_to_fillna: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of column name -> value used to fill NA before further transforms.",
    )

    target_source: str | None = Field(
        default=None,
        description="Column to derive the target from; if None, uses `target_name` as the source.",
    )
    target_name: str = Field(
        default="target", description="Name of the resulting target column."
    )
    target_strategy: TargetStrategy = Field(
        default=TargetStrategy.BINARY,
        description="Strategy for generating the target column.",
    )

    binary_rule: str | None = Field(
        default="nonzero",
        description="Binary rule: one of {'nonzero', 'threshold', 'mapping'}.",
    )
    binary_threshold: float | None = Field(
        default=None,
        description="Threshold for 'threshold' rule: target = 1 if value > threshold else 0.",
    )
    binary_mapping: dict[Any, int] = Field(
        default_factory=dict,
        description="Mapping for 'mapping' rule (e.g., {'OK':0,'WARN':1,'FAIL':1}).",
    )

    invalid_target_replacement: int | float | str | None = Field(
        default=None, description="Fallback value for invalid/unmapped target values."
    )

    @field_validator("binary_rule")
    @classmethod
    def _normalize_binary_rule(cls, v: str | None) -> str | None:
        """Normalize and validate `binary_rule`."""
        if v is None:
            return None
        v_norm = v.lower()
        if v_norm not in {"nonzero", "threshold", "mapping"}:
            raise ValueError(
                "binary_rule must be one of {'nonzero','threshold','mapping'}."
            )
        return v_norm

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DataFrameProcessorConfig:
        """
        Cross-field validation for binary strategy parameters.
        Ensures required fields are set for specific rules.
        """
        if self.target_strategy == TargetStrategy.BINARY:
            if self.binary_rule == "threshold" and self.binary_threshold is None:
                raise ValueError(
                    "binary_threshold must be set when binary_rule='threshold'."
                )
            if self.binary_rule == "mapping" and not self.binary_mapping:
                raise ValueError(
                    "binary_mapping must be provided when binary_rule='mapping'."
                )
        return self

    model_config = {
        "extra": "forbid",
        "frozen": False,  # allow editing if needed
        "validate_assignment": True,  # re-validate on assignment
        "arbitrary_types_allowed": True,  # for pandas types if ever included
    }
