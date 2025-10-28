# Preprocessing

**Function:** `preprocess_dataframe(df, parameters)`  
**Module:** `mlstream.pipelines.data_processing.dataframe_processor` (node wrapper in your project)  
**Output:** `processed_training_data` (tabular)

This node builds a target column and prepares features using a flexible config.

## Parameters (section: `params:preprocess`)

```yaml
target:
  source: Numtppd
  name: target
  strategy: binary         # binary | multiclass | regression
  rule: nonzero            # when strategy: binary → nonzero | threshold | mapping
  # threshold: 0.0         # used if rule: threshold
  # mapping: {...}         # used if rule: mapping
drop_columns:
  - Numtppd
  - Numtpbi
  - Indtppd
  - Indtpbi
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
# features: [ ... ]       # keep-only whitelist (plus target)
```

### Behavior

- **Target creation**
  - `binary` + `nonzero`: label is `1` if `source != 0`, else `0`.
  - `binary` + `threshold`: label is `1` if `source > threshold`, else `0`.
  - `binary` + `mapping`: explicit mapping dict; unmapped can fall back to `invalid_target_replacement` if provided.
  - `regression`: coerces numeric target; optionally fills invalids with `invalid_target_replacement`.

- **Column handling**
  - `drop_columns` removed **after** target creation (target itself is kept).
  - `categorical_columns` one-hot encoded with `pandas.get_dummies`.
  - `fillna` applies **before** encoding/target creation.
  - `features` (optional) restricts the final matrix to an allowlist plus the target.

### Common gotchas

```{admonition} Ensure target exists early
The `target.source` must exist in your input DataFrame. Dropping it in `drop_columns` is fine—the node drops it **after** deriving the new target.
```

### Inputs/outputs (Data Catalog keys)

- **Input:** `training_data` → e.g., an RDA dataset loaded as `mlstream.datasets.rda_dataset.RdaDataset`
- **Output:** `processed_training_data` → `pandas.CSVDataset` at `data/03_primary/processed_training_data.csv`