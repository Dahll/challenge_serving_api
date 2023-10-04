from kedro.io import AbstractDataSet
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import os


class TransformersModelDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = Path(filepath)

    def _load(self):
        return TFAutoModelForSequenceClassification.from_pretrained(str(self._filepath))

    def _save(self, model):
        self._filepath.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self._filepath))

    def _describe(self):
        return dict(filepath=str(self._filepath))
    

class TransformersTokenizerDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = Path(filepath)

    def _load(self):
        return AutoTokenizer.from_pretrained(str(self._filepath))

    def _save(self, tokenizer):
        self._filepath.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(self._filepath))

    def _describe(self):
        return dict(filepath=str(self._filepath))

