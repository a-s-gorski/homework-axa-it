<hr style="border-width:2px;border-color:darkred">
<div align="center">

<img src="./da_logo_transparent_small.gif" height=300>

<h1>DA Technical Test: ML Engineering</h1>
<br>
<br>
<hr style="border-width:2px;border-color:darkred">
</div>

## Overview

**Target Duration : 2 hours**

This technical test simulates a real-world scenario where a Data Scientist will deliver you prototype code contained in the "notebooks/modeling_starter.ipynb" notebook and your task is to transform it into a modular, maintainable, and production-ready machine learning software product. In doing so, you will need to apply software engineering best practices detailed in the **Guidelines** section and ensure the final project is fully reproducible.

You may use ML Engineering frameworks such as Kedro, ZenML, Metaflow or start from scratch **as long as your choices are justified**.

You may incorporate whichever tool you like, **as long as your choices are justified**.

You may rewrite the code in whichever way you like, **as long as your choices are justified**.

## Guidelines 

1. **Project Structure & Organization**
   - Organize the refactored code into a well-defined project layout. Consider creating directories for:
     - Data (raw data, model input, training/test data, metrics and artifacts)
     - Source code (e.g., modules for data ingestion, preprocessing, model training, evaluation, and deployment)
     - Tests (unit and integration)
     - Configuration files and scripts
     - Documentation and reporting
   - Use meaningful names for files and directories to clearly describe their content and purpose.

2. **Code Refactoring & Quality**
   - Decompose the notebook into modular parts where each module (or function) has a single objective.
   - Adhere to best coding practices and style guides. Implement tools such as `Ruff` to enforce code quality standards. 

3. **Reproducibility**
   - Provide a way to lock down all development and regular dependencies.
   - Document the steps required to set up the development environment, or consider including a Dockerfile for containerized deployment.
   - Ensure that random seeds and any non-deterministic behavior are controlled so that experiments are reproducible.

4. **Testing**
   - Write unit tests for functions you consider most critical to validate their behavior, and achieve at least 50% coverage.
   - Include an integration test to confirm that the modules work together as expected.
   - Utilize a suitable testing framework (e.g., pytest).

5. **Configuration & Hyperparameter Management**
   - Externalize configurations from the code. Use configuration files (e.g., YAML, JSON) or environment variables for parameters such as file paths, hyperparameters, and model settings.
   - Allow for easy parameter tweaking without modifying the core codebase.

6. **Documentation**
   - Provide a reasonably detailed documentation for each component, including:
     - An overview of the project’s architecture.
     - Setup and installation instructions.
     - Usage examples and command-line options.
   - Include a well-defined README that guides new users or developers through the project structure and functionality.

## Deliverables
- A refactored, modular project that replaces the original notebook.
- A comprehensive README detailing project setup, structure, usage, and testing instructions.
- Appropriate scripts or configuration files (.lock files, Dockerfile, etc.) to guarantee reproducibility.

Following these guidelines will not only demonstrate your technical aptitude in refactoring and software engineering best practices but also ensure that the resulting ML product is robust, maintainable, and reproducible, which is what we want to see.

If you have any questions you may email Pierre Adeikalam at the following address : **pierre.adeikalam[at]axa-direct.com**

Good luck!
