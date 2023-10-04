"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.train import create_pipeline as create_train_pipeline  # Assurez-vous que ceci est correct
from .pipelines.predict import create_pipeline as create_predict_pipeline  # Assurez-vous que ceci est correct

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["train"] = create_train_pipeline()
    pipelines["predict"] = create_predict_pipeline()
    return pipelines
