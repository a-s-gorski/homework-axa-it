# Data Catalog (how artifacts are stored)

Below is an example `conf/base/catalog.yml` reflecting the artifacts you listed.

```yaml
# Raw training set loaded from RDA (example custom dataset)
training_data:
  type: mlstream.datasets.rda_dataset.RdaDataset
  object_name: pg15training
  output_directory: data/02_intermediate/
  filepath: data/01_raw/pg15training.rda
  version: null

# Intermediates
converted_training_data:
  type: pandas.CSVDataset
  filepath: data/02_intermediate/pg15training.csv

processed_training_data:
  type: pandas.CSVDataset
  filepath: data/03_primary/processed_training_data.csv

# Model inputs
X_train:
  type: pandas.CSVDataset
  filepath: data/04_model_inputs/X_train.csv
y_train:
  type: pandas.CSVDataset
  filepath: data/04_model_inputs/y_train.csv
X_val:
  type: pandas.CSVDataset
  filepath: data/04_model_inputs/X_val.csv
y_val:
  type: pandas.CSVDataset
  filepath: data/04_model_inputs/y_val.csv

# Model + results (versioned)
model:
  type: pickle.PickleDataset
  filepath: data/05_training/model.pkl
  versioned: true

training_results:
  type: json.JSONDataset
  filepath: data/05_training/training_results.json
  versioned: true

# Outputs
y_pred:
  type: pandas.CSVDataset
  filepath: data/06_model_output/y_pred.csv
  versioned: false

training_metrics_json:
  type: json.JSONDataset
  filepath: data/07_reporting/training_metrics.json
  versioned: false
  save_args:
    indent: 2
    ensure_ascii: false

confusion_matrix_fig:
  type: matplotlib.MatplotlibWriter
  filepath: data/07_reporting/confusion_matrix.png
  save_args:
    dpi: 150
    bbox_inches: tight
```