"""
Module: model_registry
----------------------
This module provides a simple registry for supported machine learning model classes.
It maps string identifiers (e.g., "lightgbm", "random_forest", "logreg") to their
corresponding scikit-learn or LightGBM classifier classes.

Functions:
    retrieve_model_class(model_name: str) -> type:
        Returns the model class associated with the given name.

Usage example:
    model_cls = retrieve_model_class("lightgbm")
    model = model_cls()
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_TYPE = (
    type[LGBMClassifier] | type[RandomForestClassifier] | type[LogisticRegression]
)

ALLOWED_MODELS: dict[str, MODEL_TYPE] = {
    "lightgbm": LGBMClassifier,
    "random_forest": RandomForestClassifier,
    "logreg": LogisticRegression,
}


def retrieve_model_class(model_name: str) -> MODEL_TYPE:
    """Get the model class from its name."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model type: {model_name}")
    return ALLOWED_MODELS[model_name]
