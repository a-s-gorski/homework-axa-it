import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from mlstream.pipelines.reporting.nodes import (
    confusion_matrix_figure,
    metrics_json,
    predict_labels,
)


class DummyModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        n = len(X)
        return self._preds[:n]


def test_predict_labels_aligns_index_and_series_type():
    X = pd.DataFrame({"f1": [0.1, 0.2]}, index=[10, 11])
    model = DummyModel([1, 0])
    s = predict_labels(model, X)
    assert isinstance(s, pd.Series)
    assert s.index.tolist() == [10, 11]
    assert s.tolist() == [1, 0]


def test_predict_labels_series_name_suffix():
    X = pd.Series([0.1, 0.2, 0.3], name="features", index=[5, 6, 7])
    model = DummyModel([0, 1, 1])
    s = predict_labels(model, X)
    assert s.name == "features_pred"
    assert s.index.tolist() == [5, 6, 7]
    assert s.tolist() == [0, 1, 1]


def test_metrics_json_binary_values():
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    m = metrics_json(y_true, y_pred)

    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3, rel=1e-6)


def _extract_cm_from_axes(ax):
    img = ax.images[0]
    arr = np.asarray(img.get_array())
    return arr


def test_confusion_matrix_figure_normalized_true(capsys):
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])

    fig = confusion_matrix_figure(y_true, y_pred, labels=[0, 1], normalize="true")
    capsys.readouterr()

    ax = fig.axes[0]
    cm_plot = _extract_cm_from_axes(ax)
    expected = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert np.allclose(cm_plot, expected, atol=1e-8)


def test_confusion_matrix_respects_label_order(capsys):
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])

    fig = confusion_matrix_figure(y_true, y_pred, labels=[1, 0], normalize=None)
    capsys.readouterr()

    ax = fig.axes[0]
    cm_plot = _extract_cm_from_axes(ax)
    expected = np.array([[1, 1], [0, 2]])
    assert np.array_equal(cm_plot, expected)
