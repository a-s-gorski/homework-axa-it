from kedro.pipeline import Node, Pipeline

from .nodes import confusion_matrix_figure, metrics_json, predict_labels


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                func=predict_labels,
                inputs=["model", "X_val"],
                outputs="y_pred",
                name="predict_labels_node",
            ),
            Node(
                func=metrics_json,
                inputs=["y_val", "y_pred"],
                outputs="training_metrics_json",
                name="metrics_json_node",
            ),
            Node(
                func=confusion_matrix_figure,
                inputs={
                    "y_true": "y_val",
                    "y_pred": "y_pred",
                    "labels": "params:cm_labels",
                    "normalize": "params:cm_normalize",
                },
                outputs="confusion_matrix_fig",
                name="confusion_matrix_fig_node",
            ),
        ]
    )
