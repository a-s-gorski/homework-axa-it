from kedro.pipeline import Node, Pipeline

from .nodes import split_train_test, train_or_search_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_train_test,
                inputs=["processed_training_data", "params:model_options"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data_node",
            ),
            Node(
                func=train_or_search_model,
                inputs=["X_train", "y_train", "X_val", "y_val", "params:modeling"],
                name="train_or_search_model_node",
                outputs=["model", "training_results"],
            ),
        ]
    )
