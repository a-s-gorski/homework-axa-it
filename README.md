# Mlstram - Kedro based package for practical Data Science Development

This repository contains a refactored and modular machine learning pipeline for data processing, model training and evaluation automatically.


The solution is implemented using **Kedro**, which provides standardized project structure, modular pipelines, and automatic artifact management. The project is designed to support collaboration between Data Scientists, Developers, and Technical Writers, ensuring reproducibility, code quality, and maintainable documentation.

## Project Overview

**Key Features**

- Modular pipeline with reusable nodes for data ingestion, preprocessing, model training, and evaluation

- Config-driven experimentation (no code edits required for new experiments)

- Pre-commit hooks enforcing linting, formatting, type checking, and security scans

- Unit, integration, and end-to-end tests with CI integration

- Auto-generated documentation with Sphinx and deployment to GitHub Pages

- Custom Kedro dataset for .rda file handling

- Environment reproducibility using pyproject.toml and optional virtual environment setup


## For Data Scientists

### 1. Environment Setup
You can use either `venv` or `conda`.

**Using venv:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

**Using conda:**
```bash
conda create -n kedro-env python=3.11
conda activate kedro-env
pip install -e .[dev]
```

This installs the project in editable mode (`-e .`) so that any code or configuration changes are immediately available.


### 2. Project Configuration
All configuration files are located under `conf/base/`:

- `catalog.yml`: defines datasets, their types, file paths, and optional versioning  
- `parameters.yml`: stores model hyperparameters and global constants  
- `parameters/<pipeline>.yml`: stores pipeline-specific parameters (e.g., selected columns, Optuna search space, etc.)

You can run the pipeline directly with:
```bash
kedro run
```

To run only specific nodes:
```bash
kedro run --node=preprocess_data
```

To modify experiment settings, simply update the relevant YAML file — no code changes are required.

### 3. Experiment Reproducibility
Random seeds and deterministic options are stored in configuration files.  
Artifacts (trained models, metrics, datasets) are versioned and saved automatically by Kedro.


## For Technical Writers

### 1. Building Documentation
Documentation is built using **Sphinx** (configured via Kedro).

To generate HTML docs:
```bash
cd docs
make html
```

The generated site will appear in:
```
docs/build/html/index.html
```

On Linux you can see the results locally using:
```bash
cd docs
open docs/build/html/index.html
```

After each merge request the docs will be automatically pushed to Github Pages and the result will be visible here:
https://a-s-gorski.github.io/homework-axa-it/



### 3. Docstring Standards
All modules use typed docstrings (NumPy style) and are automatically parsed by Sphinx’s autodoc extension, ensuring that new code is reflected in documentation builds without manual intervention.

---


## For Developers

### 1. Code Quality and Pre-commit Hooks
The project enforces strict code quality through **pre-commit**.  
Install hooks once after environment setup:

```bash
pre-commit install
```

The hooks run automatically before each commit and include:
- **Ruff** – linting and import sorting  
- **Black** – code formatting  
- **MyPy** – static type checking  
- **Bandit** – security scanning  

You can also run them manually:
```bash
pre-commit run --all-files
```

### 2. Testing
Run all unit and integration tests with:
```bash
pytest --cov=src
```

Coverage reports are generated automatically.  
Test datasets are stored under `data/test/`.

### 3. Continuous Integration (CI/CD)
CI is configured through **GitHub Actions**. Each pull request or push triggers:
- Environment setup  
- Static analysis (linting, typing, security)  
- Unit and integration tests with coverage reports  

Future CI/CD extensions (MLflow integration, environment-specific model promotion) can be added easily.

### 4. Git Workflow
- Create feature branches from `main`  
- Commit frequently with meaningful messages  
- Ensure all hooks and tests pass before pushing  
- Submit pull requests for review
- CI must pass for merge approval

---

## Project Structure

```
├── conf/                     # Configuration files (YAML)
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters_data_processing.yml
│   │   ├── parameters_data_science.yml
│   │   ├── parameters_data_reporting.yml
├── data/                     # Raw, intermediate, and output data
├── docs/                     # Sphinx documentation
├── scripts/                  # Utility scripts
├── src/                      # Main Python source code
│   ├── mlstream
│   │   ├── pipelines/
│   │   ├── nodes/
│   │   ├── datasets/
│   │   └── __init__.py
├── tests/                    # Unit and integration tests
├── pyproject.toml            # Project metadata and dependencies
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .github/workflows/ci.yml  # CI configuration
└── README.md
```

---

## Future Idead
- Integration with **MLflow** for experiment tracking and model versioning  
- Environment-specific configuration (e.g., staging/production)  
- Canary or batch deployment using Docker/Fargate  
- Drift detection and retraining automation  
