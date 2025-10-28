from kedro.pipeline import Node, Pipeline

from .nodes import preprocess_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=preprocess_dataframe,
                inputs=["training_data", "params:preprocess"],
                outputs="processed_training_data",
                name="preprocess_dataframe_node",
            )
        ]
    )
