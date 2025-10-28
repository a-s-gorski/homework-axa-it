# Data science

This stage splits data and either trains directly or runs Optuna search (if enabled).

## Node: `split_train_test`

**Signature:** `split_train_test(df, parameters)`  
**Parameters (`params:model_options`):**
```yaml
target_column: "target"
test_size: 0.3
random_state: 42
stratify: true
```

- Splits `processed_training_data` into `X_train`, `X_val`, `y_train`, `y_val`.
- If `stratify` is `true`, uses `y` for stratification.

## Node: `train_or_search_model`

**Signature:** `train_or_search_model(X_train, y_train, X_val, y_val, parameters_modeling)`  
**Parameters (`params:modeling`):**

### Direct training (search disabled)
```yaml
model:
  type: lightgbm          # lightgbm | random_forest | logreg | xgboost (if supported in your resolver)
  params:
    objective: binary
    n_estimators: 120
    learning_rate: 0.05
    max_depth: 6
    num_leaves: 31
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8
    n_jobs: -1
    verbosity: -1
metric: accuracy
search:
  enabled: false
```

### Optuna search (alternative)
```yaml
model:
  type: lightgbm
  params:
    objective: binary
    n_jobs: -1
    verbosity: -1
metric: accuracy
search:
  enabled: true
  n_trials: 20
  direction: maximize
  test_size: 0.2
  space:
    n_estimators: {type: int, low: 50, high: 300}
    learning_rate: {type: float, low: 0.001, high: 0.3, log: true}
    max_depth: {type: int, low: 3, high: 10}
    num_leaves: {type: int, low: 20, high: 150}
    min_child_samples: {type: int, low: 5, high: 100}
    subsample: {type: float, low: 0.5, high: 1.0}
    colsample_bytree: {type: float, low: 0.5, high: 1.0}
```

### Outputs
- `model`: trained estimator (pickled via Data Catalog).
- `training_results`: info dict (e.g., `best_params`, `best_value`, `n_trials`, `mode`).