"""
Module: evaluation
------------------
Provides helper functions for model evaluation and visualization of classification results.

Includes:
- Generating predictions aligned to test data indices.
- Computing common classification metrics (accuracy, precision, recall, F1).
- Creating confusion matrix plots with optional normalization.

Dependencies
------------
- pandas
- scikit-learn
- matplotlib

Functions
---------
predict_labels(best_model, X_test)
    Generate predicted class labels as a pandas Series aligned with test data.
metrics_json(y_true, y_pred)
    Compute accuracy, precision, recall, and F1 score as a serializable dictionary.
confusion_matrix_figure(y_true, y_pred, labels=None, normalize=None)
    Plot and return a confusion matrix as a Matplotlib Figure.

Usage example
-------------
>>> from sklearn.ensemble import RandomForestClassifier
>>> from mypkg.evaluation import predict_labels, metrics_json, confusion_matrix_figure
>>> model = RandomForestClassifier().fit(X_train, y_train)
>>> preds = predict_labels(model, X_test)
>>> metrics = metrics_json(y_test, preds)
>>> fig = confusion_matrix_figure(y_test, preds, normalize="true")
>>> fig.show()
"""

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_labels(best_model, X_test) -> pd.Series:
    """
    Generate class predictions as a pandas Series aligned to the index of X_test.

    Parameters
    ----------
    best_model : object
        A trained classifier with a `.predict()` method.
    X_test : array-like or pandas.DataFrame or pandas.Series
        Test feature data for which predictions are made.

    Returns
    -------
    pandas.Series
        Predicted class labels, aligned to `X_test`'s index (if available).
        The name is set to `<X_test.name>_pred` when `X_test` is a named Series.

    Examples
    --------
    >>> preds = predict_labels(model, X_test)
    >>> preds.head()
    0    1
    1    0
    dtype: int64
    """
    preds = best_model.predict(X_test)
    index = getattr(X_test, "index", None)

    s = pd.Series(preds, index=index)
    if isinstance(X_test, pd.Series) and X_test.name:
        s.name = f"{X_test.name}_pred"

    return s


def metrics_json(y_true, y_pred) -> dict[str, float]:
    """
    Compute basic classification metrics and return them as a JSON-serializable dict.

    Parameters
    ----------
    y_true : array-like
        Ground truth target labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "accuracy": Accuracy score.
        - "precision": Precision score.
        - "recall": Recall score.
        - "f1": F1 score.

    Notes
    -----
    Division-by-zero cases are handled gracefully with `zero_division=0`.

    Examples
    --------
    >>> metrics_json([0, 1, 1, 0], [0, 1, 0, 0])
    {'accuracy': 0.75, 'precision': 1.0, 'recall': 0.5, 'f1': 0.67}
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_figure(
    y_true,
    y_pred,
    labels: Optional[Sequence] = None,
    normalize: Optional[str] = None,
):
    """
    Create a confusion matrix visualization as a Matplotlib Figure.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target labels.
    y_pred : array-like
        Predicted target labels.
    labels : sequence, optional
        List of label names for display order in the confusion matrix.
    normalize : {'true', 'pred', 'all'}, optional
        Normalization mode for confusion matrix. Pass None for raw counts.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure object containing the confusion matrix plot.

    Examples
    --------
    >>> fig = confusion_matrix_figure(y_test, y_pred, normalize='true')
    >>> fig.show()
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(
        "Confusion Matrix" + (f" (normalize={normalize})" if normalize else "")
    )
    fig.tight_layout()
    return fig
