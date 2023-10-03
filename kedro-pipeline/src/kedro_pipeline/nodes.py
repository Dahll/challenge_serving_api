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

    X_train, X_test, y_train, y_test = train_test_split(data['review'],
                                                    data['label'],
                                                    test_size = parameters["test_size"], 
                                                    random_state = parameters["random_state"])

    logger = logging.getLogger(__name__)
    logger.info(f'Train: {len(X_train)}')
    logger.info(f'Test: {len(X_test)}')

    #print(f'Train: {len(X_train)}')
    #print(f'Test: {len(X_test)}')


    #data_train = data.sample(
    #    frac=parameters["train_fraction"], random_state=parameters["random_state"]
    #)
    #data_test = data.drop(data_train.index)

    #X_train = data_train.drop(columns=parameters["target_column"])
    #X_test = data_test.drop(columns=parameters["target_column"])
    #y_train = data_train[parameters["target_column"]]
    #y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


def make_training(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
):
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """

    #Load tokenizer
    logger = logging.getLogger(__name__)
    logger.info(f'Load tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    preprocessed_train = tokenizer(X_train.to_list(), return_tensors="np", padding=True)
    preprocessed_test = tokenizer(X_test.to_list(), return_tensors="np", padding=True)

    # Create label list
    logger.info(f'Create label list')
    labels_train = np.array(y_train)  
    labels_test = np.array(y_test)

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
          batch_size=4, 
          epochs=2)
    
    
    return model
    test_predictions = model.predict(dict(preprocessed_test))['logits']
    test_probabilities = tf.nn.softmax(test_predictions)
    y_pred = np.argmax(test_probabilities, axis=1)

    #X_train_numpy = X_train.to_numpy()
    #X_test_numpy = X_test.to_numpy()

    #squared_distances = np.sum(
    #    (X_train_numpy[:, None, :] - X_test_numpy[None, :, :]) ** 2, axis=-1
    #)
    #nearest_neighbour = squared_distances.argmin(axis=0)
    #y_pred = y_train.iloc[nearest_neighbour]
    #y_pred.index = X_test.index

    return y_pred


def report_accuracy(X_test: pd.Series, y_test: pd.Series, bert_model):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    preprocessed_test = tokenizer(X_test.to_list(), return_tensors="np", padding=True)

    test_predictions = bert_model.predict(dict(preprocessed_test))['logits']
    test_probabilities = tf.nn.softmax(test_predictions)
    y_pred = np.argmax(test_probabilities, axis=1)

    accuracy = (y_pred == y_test).sum() / len(y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Model Version", 25)
