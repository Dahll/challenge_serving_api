"""
This is a boilerplate pipeline
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from ..nodes import make_training, inference, split_data, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["amazon_cells_labelled", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="Split",
            ),
            node(
                func=make_training,
                inputs=["X_train", "X_test", "y_train", "y_test", "parameters"],
                outputs=["bert_model", "tokenizer"],
                name="Train",
            ),
        ]
    )
