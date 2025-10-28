import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
from matplotlib.figure import Figure

from mlstream.pipelines.data_processing.pipeline import create_pipeline as dp_create
from mlstream.pipelines.data_science.pipeline import create_pipeline as ds_create
from mlstream.pipelines.reporting.pipeline import create_pipeline as rp_create


@pytest.mark.e2e
def test_full_kedro_pipeline_end_to_end(
    raw_training_df,
    preprocess_params,
    ds_split_params,
    modeling_params,
    reporting_params,
):
    dp = dp_create()
    ds = ds_create()
    rp = rp_create()
    full: Pipeline = dp + ds + rp

    catalog = DataCatalog(
        {
            "training_data": MemoryDataset(raw_training_df),
            "params:preprocess": MemoryDataset(preprocess_params),
            "params:model_options": MemoryDataset(ds_split_params),
            "params:modeling": MemoryDataset(modeling_params),
            "params:cm_labels": MemoryDataset(reporting_params["cm_labels"]),
            "params:cm_normalize": MemoryDataset(reporting_params["cm_normalize"]),
        }
    )

    out = SequentialRunner().run(full, catalog=catalog)

    expected_keys = {
        "training_results",
        "training_metrics_json",
        "confusion_matrix_fig",
    }
    assert expected_keys.issubset(out.keys())

    def _val(v):
        return v.load() if hasattr(v, "load") else v

    training_results = _val(out["training_results"])
    training_metrics = _val(out["training_metrics_json"])
    cm_fig = _val(out["confusion_matrix_fig"])

    assert isinstance(training_results, dict)
    assert "best_params" in training_results
    assert "mode" in training_results

    assert isinstance(training_metrics, dict)
    assert {"accuracy", "precision", "recall", "f1"}.issubset(training_metrics.keys())
    for k in ["accuracy", "precision", "recall", "f1"]:
        assert isinstance(training_metrics[k], float)

    assert isinstance(cm_fig, Figure)
