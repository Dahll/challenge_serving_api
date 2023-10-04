import pathlib
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[1])+ "/kedro-pipeline/src/kedro_pipeline/pipelines")
print(str(Path(__file__).resolve().parents[1])+ "/kedro-pipeline/src/kedro_pipeline/pipelines")


from nodes import inference

import numpy as np
import pandas as pd
import mlflow
import mlflow.tracking
import tensorflow as tf

from kedro.framework.context import KedroContext
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import sklearn
from sklearn import metrics
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


def load_artefacts():
    #Select path where mlflow is
    path_uri = "file://" + str(Path(__file__).resolve().parents[1])+ "/kedro-pipeline/mlruns"
    mlflow.set_tracking_uri(path_uri)
    
    
    #Find last working run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("kedro_pipeline")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs = runs[runs["status"] == "FINISHED"]
    run_id = runs.sort_values(by='start_time', ascending=False).iloc[0]["run_id"]
    
    #Create destination dir
    destination_dir = os.path.join(pathlib.Path().absolute(), "data")
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    
    #Download artifacts
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, "bert_trained_model", destination_dir)
    client.download_artifacts(run_id, "tokenizer", destination_dir)
    
    #Load artifacts
    model = TFAutoModelForSequenceClassification.from_pretrained(destination_dir + "/" + "bert_trained_model")
    tokenizer = AutoTokenizer.from_pretrained(destination_dir + "/" + "tokenizer")
    
    return model, tokenizer


model, tokenizer = load_artefacts()

app = FastAPI()

class Sentence(BaseModel):
    sentence: str

@app.post("/v1/inference/sentiment_analysis")
def sentiment_analysis(item: Sentence):
    data = {
    'Phrase': [item.sentence]
    }

    df = pd.DataFrame(data)
    res = inference(df, model, tokenizer).to_numpy()

    if np.argmax(res):
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    result = {
        "sentiment": sentiment,
        "confidence": round(float(res[0][np.argmax(res)]), 2)
    }
    return result