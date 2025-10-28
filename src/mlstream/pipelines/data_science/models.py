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
