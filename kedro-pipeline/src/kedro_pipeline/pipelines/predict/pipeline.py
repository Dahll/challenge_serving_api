"""
This is a boilerplate pipeline
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from ..nodes import inference, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=inference,
                inputs=["X_test", "bert_model", "tokenizer"],
                outputs="y_pred",
                name="Infer",
            ),
            node(
                func=evaluate,
                inputs=["y_pred", "y_test"],
                outputs="accuracy",
                name="Evaluate",
            ),
        ]
    )
