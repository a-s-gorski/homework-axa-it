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
    """Return class predictions aligned to X_test's index."""
    preds = best_model.predict(X_test)
    index = getattr(X_test, "index", None)

    s = pd.Series(preds, index=index)
    if isinstance(X_test, pd.Series) and X_test.name:
        s.name = f"{X_test.name}_pred"

    return s


def metrics_json(y_true, y_pred) -> dict[str, float]:
    """Return basic classification metrics as a JSON-serializable dict."""
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
    """Return a Matplotlib Figure with the confusion matrix."""
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
