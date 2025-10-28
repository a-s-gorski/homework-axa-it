# tests/pipelines/data_processing/test_dataframe_processor.py
import logging
import pandas as pd
import pytest

from mlstream.pipelines.data_processing.dataframe_processor import (
    DataFrameProcessor,
)
from mlstream.pipelines.data_processing.dataframe_processor_config import (
    DataFrameProcessorConfig,
    TargetStrategy,
)

# ----------------------------- Config validation -----------------------------

def test_config_threshold_rule_requires_threshold():
    with pytest.raises(ValueError, match="binary_threshold must be set"):
        DataFrameProcessorConfig(
            target_strategy=TargetStrategy.BINARY,
            binary_rule="threshold",
            # binary_threshold missing on purpose
        )

def test_config_mapping_rule_requires_mapping():
    with pytest.raises(ValueError, match="binary_mapping must be provided"):
        DataFrameProcessorConfig(
            target_strategy=TargetStrategy.BINARY,
            binary_rule="mapping",
            binary_mapping={},  # empty on purpose
        )

@pytest.mark.parametrize("rule", ["nonzero", "threshold", "mapping"])
def test_config_binary_rule_normalization(rule):
    cfg = DataFrameProcessorConfig(
        target_strategy=TargetStrategy.BINARY,
        binary_rule=rule.upper(),  # should normalize to lowercase
        binary_threshold=0.5 if rule == "threshold" else None,
        binary_mapping={"OK": 0, "FAIL": 1} if rule == "mapping" else {},
    )
    assert cfg.binary_rule == rule


# ------------------------------- Helper factory ------------------------------

def make_processor(**overrides) -> DataFrameProcessor:
    """
    Build a DataFrameProcessor while avoiding duplicate kwargs.
    We rely on Pydantic defaults and only pass what the test needs.
    """
    base = dict(
        # rely on model defaults for everything else
    )
    base.update(overrides)
    cfg = DataFrameProcessorConfig(**base)
    return DataFrameProcessor(cfg)


# ---------------------------- Binary target rules ----------------------------

def test_binary_nonzero_encoding():
    df = pd.DataFrame({"x": [0, 1, -2, None]})
    p = make_processor(target_source="x", binary_rule="nonzero")
    out = p.process(df)
    assert out["target"].tolist() == [0, 1, 1, 0]

def test_binary_threshold_encoding():
    df = pd.DataFrame({"x": [0, 0.5, 1.1, "2.0", None]})
    p = make_processor(
        target_source="x", binary_rule="threshold", binary_threshold=1.0
    )
    out = p.process(df)
    assert out["target"].tolist() == [0, 0, 1, 1, 0]

def test_binary_mapping_with_fallback_and_invalid_replacement():
    df = pd.DataFrame({"status": ["OK", "WARN", "UNKNOWN", None]})
    p = make_processor(
        target_source="status",
        binary_rule="mapping",
        binary_mapping={"OK": 0, "WARN": 1, "FAIL": 1},
        invalid_target_replacement=0,
    )
    out = p.process(df)
    assert out["target"].tolist() == [0, 1, 0, 0]


# ---------------------------- Regression target ------------------------------

def test_regression_target_with_invalid_replacement():
    df = pd.DataFrame({"y": ["1.2", "bad", 3.0, None]})
    p = make_processor(
        target_strategy=TargetStrategy.REGRESSION,
        target_source="y",
        invalid_target_replacement=-1.0,
    )
    out = p.process(df)
    assert out["target"].tolist() == [1.2, -1.0, 3.0, -1.0]


# ------------------------- FillNA before target logic ------------------------

def test_fillna_happens_before_target_creation_with_threshold():
    df = pd.DataFrame({"score": [None, 2, None]})
    p = make_processor(
        columns_to_fillna={"score": 5},
        target_source="score",
        binary_rule="threshold",
        binary_threshold=3,
    )
    out = p.process(df)
    assert out["target"].tolist() == [1, 0, 1]


# ---------------------------- One-hot & dropping -----------------------------

def test_one_hot_encoding_drops_original_and_joins_dummies():
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(["A", "B", "C", "A"]),
            "x": [1, 2, 3, 4],
        }
    )
    p = make_processor(
        target_strategy=TargetStrategy.REGRESSION,
        target_source="x",
        columns_to_onehotencode=["cat"],
    )
    out = p.process(df)
    assert "cat" not in out.columns
    assert set(col for col in out.columns if col.startswith("cat_")) == {"cat_B", "cat_C"}
    assert "target" in out.columns
    assert out["target"].tolist() == [1, 2, 3, 4]

def test_drop_columns_protects_target():
    df = pd.DataFrame({"y": [0, 1, 0]})
    p = make_processor(
        target_source="y",
        columns_to_drop=["y", "target"],
    )
    out = p.process(df)
    assert "target" in out.columns
    assert "y" not in out.columns

def test_existing_target_without_source_is_preserved():
    df = pd.DataFrame({"target": [7, 8, 9], "other": [1, 2, 3]})
    p = make_processor(
        target_source=None,
        target_name="target",
        target_strategy=TargetStrategy.BINARY,
    )
    out = p.process(df)
    assert out["target"].tolist() == [7, 8, 9]


# ------------------------------- Validation logs -----------------------------

def test_validate_logs_missing_and_extra_columns_and_dtype_warnings(caplog):
    df = pd.DataFrame(
        {
            "to_drop": [1, 2],
            "to_fill": [None, 2],
            "to_onehot": ["a", "b"],  # not categorical -> should warn
            "extra_col": [0, 0],      # extra -> should warn
            "source": [0, 1],         # target source
        }
    )

    p = make_processor(
        columns_to_drop=["to_drop"],
        columns_to_fillna={"to_fill": 0, "missing_col": 0},  # missing_col is missing
        columns_to_onehotencode=["to_onehot"],
        target_source="source",
        target_name="y",
    )

    caplog.clear()
    caplog.set_level(logging.WARNING)  # capture warnings/errors
    assert p._validate(df) is True

    # Missing columns error
    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    assert any("Missing columns in DataFrame" in rec.message and "missing_col" in rec.message for rec in errors)

    # Extra columns warning (extra_col)
    warns = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("Extra columns in DataFrame" in rec.message and "extra_col" in rec.message for rec in warns)

    # Non-categorical one-hot warning
    assert any("is not categorical (will still be one-hot encoded)" in rec.message for rec in warns)
