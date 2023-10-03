from kedro.io import AbstractDataSet
from transformers import TFAutoModelForSequenceClassification
from pathlib import Path
import os


class TransformersModelDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = Path(filepath)
        if not os.path.exists(self._filepath):
            os.mkdir(self._filepath)

    def _load(self):
        return TFAutoModelForSequenceClassification.from_pretrained(str(self._filepath))

    def _save(self, model):
        model.save_pretrained(str(self._filepath))

    def _describe(self):
        return dict(filepath=str(self._filepath))
