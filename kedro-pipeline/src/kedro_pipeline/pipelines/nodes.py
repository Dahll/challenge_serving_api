"""
This is a boilerplate pipeline
generated using Kedro 0.18.13
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow import sklearn
from sklearn import metrics
from datetime import datetime

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    X_train, X_test, y_train, y_test = train_test_split(data[parameters["X_column_name"]],
                                                    data[parameters["y_column_name"]],
                                                    test_size = parameters["test_size"], 
                                                    random_state = parameters["random_state"])

    logger = logging.getLogger(__name__)
    logger.info(f'Train: {len(X_train)}')
    logger.info(f'Test: {len(X_test)}')

    return X_train, X_test, y_train, y_test


def make_training(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, parameters: Dict[str, Any]
):
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        model: Prediction of the target variable.
    """
    #print(X_train.iloc[:, 0].tolist())
    #Load tokenizer
    logger = logging.getLogger(__name__)
    logger.info(f'Load tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    preprocessed_train = tokenizer(X_train.iloc[:, 0].tolist(), return_tensors="np", padding=True)
    preprocessed_test = tokenizer(X_test.iloc[:, 0].tolist(), return_tensors="np", padding=True)

    # Create label list
    logger.info(f'Create label list')
    labels_train = y_train.to_numpy()
    labels_test = y_test.to_numpy()

    # Load pre-trained model
    logger.info(f'Load pretrained model')
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=Adam(5e-6), loss=loss_function, metrics=['accuracy'])

    # Fit the model
    logger.info(f'Fit the model')

    model.fit(dict(preprocessed_train), 
          labels_train, 
          validation_data=(dict(preprocessed_test), labels_test),
          batch_size=parameters["batch_size"], 
          epochs=parameters["epochs"])
    
    
    return model, tokenizer


def inference(X_test: pd.Series, model, tokenizer):
    """Make one inference.

    Args:
        X_test: Predicted target.
        model: The model to use.
        tokenizer : 
    
    Returns:
        y_pred
    """
    preprocessed_test = tokenizer(X_test.iloc[:, 0].tolist(), return_tensors="np", padding=True)

    test_predictions = model.predict(dict(preprocessed_test))['logits']
    y_pred = tf.nn.softmax(test_predictions)
    
    return pd.DataFrame(y_pred)


def evaluate(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """

    

    y_pred = y_pred.to_numpy()
    y_pred = np.argmax(y_pred, axis=1)
    
    y_test = y_test.iloc[:, 0].to_numpy()

    accuracy = accuracy_score(y_pred, y_test)
    
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Model Version", 25)
    
    return accuracy