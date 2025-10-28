# Parameters (how to configure)

All behavior is driven from Kedro `parameters`. Below is a consolidated example you can place under `conf/base/parameters.yml` (override in `conf/local/parameters.yml` as needed).

```yaml
# --- Preprocessing ---
preprocess:
  target:
    source: Numtppd
    name: target
    strategy: binary
    rule: nonzero
  drop_columns: [Numtppd, Numtpbi, Indtppd, Indtpbi]
  categorical_columns:
    - CalYear
    - Gender
    - Type
    - Category
    - Occupation
    - SubGroup2
    - Group2
    - Group1
  # fillna: {Occupation: "UNK", Bonus: 0, Age: 0}
  # invalid_target_replacement: 0
  # features: [...]

# --- Data split options ---
model_options:
  target_column: "target"
  test_size: 0.3
  random_state: 42
  stratify: true

# --- Modeling ---
modeling:
  model:
    type: lightgbm
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
    n_trials: 20
    direction: maximize
    test_size: 0.2
    space: {}

# --- Evaluation ---
cm_labels: null
cm_normalize: null
```