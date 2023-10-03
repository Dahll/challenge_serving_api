"""
This is a boilerplate pipeline
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_training, report_accuracy, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["amazon_cells_labelled", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=make_training,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="bert_model",
                name="make_training",
            ),
            node(
                func=report_accuracy,
                inputs=["X_test", "y_test", "bert_model"],
                outputs=None,
                name="report_accuracy",
            ),
        ]
    )
