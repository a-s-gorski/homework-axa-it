# Evaluation & Reporting

Utilities live in the evaluation module; a Kedro pipeline wires them as nodes.

## Functions

### `predict_labels(best_model, X_test) -> pd.Series`
- Runs `model.predict(X_test)`, returns a `Series` aligned to `X_test.index`.

### `metrics_json(y_true, y_pred) -> dict[str, float]`
- Computes `accuracy`, `precision`, `recall`, `f1` (binary classification); uses `zero_division=0`.

### `confusion_matrix_figure(y_true, y_pred, labels=None, normalize=None) -> Figure`
- Builds a confusion matrix plot via `sklearn.metrics.ConfusionMatrixDisplay`.
- `normalize` can be `null`, `"true"`, `"pred"`, or `"all"`.

## Parameters (evaluation)

```yaml
cm_labels: null          # e.g. [0, 1] or ["neg", "pos"]
cm_normalize: null       # null | true | pred | all
```

## Outputs
- **`y_pred`** → `data/06_model_output/y_pred.csv`
- **`training_metrics_json`** → `data/07_reporting/training_metrics.json`
- **`confusion_matrix_fig`** → `data/07_reporting/confusion_matrix.png`