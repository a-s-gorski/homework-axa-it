# Solution README

This solution provides my explanations and design decisions.

0. **Notebook Observations**

- The requirements file lacks specified package versions. Installing dependencies directly within the notebook is generally not a good practice (except in environments like Google Colab). Additionally, the imports are unsorted — I’d recommend using isort for consistent import ordering.

- The dataset is downloaded repeatedly in each run. It would be more efficient to check whether the files already exist before re-downloading them.

- I don’t fully understand why all minority classes were converted to 0. These classes may represent meaningful real-world patterns and should be further examined — or at least require a clear business justification for their removal.

- Disabling logging warnings conceals potential issues with data or model training and is generally considered a code smell.

- Several columns are dropped without clear explanation. Although they may contain many missing values, I’d prefer to see a correlation plot or feature importance analysis to justify these decisions.

- One-hot encoding the Year feature doesn’t seem appropriate, even with low cardinality, as it discards temporal trends inherent in the data.

- The resulting dataset becomes very high-dimensional. I would consider applying dimensionality reduction techniques, removing highly correlated features, or combining similar ones.

- A simple train/validation split is insufficient. I’d recommend reserving a hidden test set for final performance reporting, using the validation set for hyperparameter optimization (e.g., via Optuna). Additionally, stratified splitting should be considered to preserve class distribution.

- Using accuracy alone for heavily imbalanced data is misleading. Metrics that better reflect the balance between precision and recall, such as the F1 score, are more appropriate. Cross-validation could also improve robustness for such datasets.

- The confusion matrix shows that the model tends to predict the majority class. While this provides some insight, I’d perform a deeper evaluation when the model underperforms — for example, using SHAP values to identify which features have the most influence on predictions.

- I do not think that optuna search has to be run every single time, instead I have made it optional, where if it is disabled, model training is run instead.

1. **Project Structure & Organization**

I selected Kedro as the framework because it enforces modular, reproducible pipelines, clear data lineage, and configuration separation — fully aligned with the task requirements.
Kedro provides standard directories for data/, src/, conf/, and tests/, with automatic artifact versioning and pipeline visualization.

This project is implemented as a monorepo, which simplifies maintenance for this scale and use case.
A custom Kedro dataset was added for loading .rda files. Common developer scripts are placed in scripts/, and a GitHub Actions pipeline handles CI.
Pre-commit hooks (Ruff, Black, MyPy, Bandit) ensure that Data Scientists and Developers commit only compliant code.


2. **Code Refactoring & Quality**
The notebook was decomposed into modular pipeline nodes with clear single responsibilities.
Column selection, preprocessing, and model training are parameterized through config files, enabling experimentation without code changes.

Data validation runs before preprocessing to prevent downstream errors.
The preprocessing logic uses OOP and inheritance to facilitate extensibility (e.g., future TensorDataProcessor).
CodePreprocessor was defined as a Pydantic model for schema enforcement and safe configuration loading.

Tools used:

- Black – code formatting

- MyPy – static typing

- Bandit – security scanning

- Ruff – linting and import sorting
- This setup standardizes quality checks and minimizes debugging effort. All modules include type-annotated docstrings following a consistent style.

3. **Reproducibility**
Kedro’s pyproject.toml manages dependencies. They are grouped by use case (dev, test, docs) for modular installation.
For development libraries, I allow backward-compatible versions to avoid dependency conflicts.
production builds would pin versions explicitly (via lock file or Docker image).

All seeds and random states are configured through YAML, ensuring experiment reproducibility.
If the system scales to multi-library integration, I would consider Poetry for easier namespace and subpackage management.

4. **Testing**
Tests are implemented with pytest:

- Unit tests validate transformations and schema expectations, achieving ~70% coverage.

- Integration tests verify pipeline connectivity and data flow between nodes.

- End-to-end tests confirm the full training workflow.

- CI integration in GitHub Actions provides coverage reports and fail-fast checks for every push.

5. **Configuration & Hyperparameter Management**
All parameters and file paths are externalized into YAML configs:

- catalog.yml defines data inputs/outputs and enables versioned storage (local or remote, e.g., S3).

- parameters.yml and pipeline-specific configs define hyperparameters, column lists, and other runtime options.
- This design allows Data Scientists to modify experiments entirely through configuration, without touching core code.

6. **Documentation**

The README.md explains setup and usage from the perspective of developers, data scientists, and DevOps engineers.

In addition, Sphinx documentation (pre-configured via Kedro) generates API references and has been deployed to GitHub Pages.

This approach keeps the README concise while ensuring scalable documentation as the project grows.

In previous work I’ve also used MkDocs and Docusaurus. Sphinx was selected here for native Kedro integration.


Planned enhancements include:

- MLflow integration for experiment tracking and model versioning, allowing model retrieval via configuration in production.

- Environment-specific configurations (e.g., for staging and production).

- CI-level integration tests before merge for stronger release gating.

- Additional steps in CI pipeline to build package and push to Github.