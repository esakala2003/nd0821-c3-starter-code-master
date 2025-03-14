"""
Module for unit test on the model
Author: Emmanuel Sakala
Date: 13/03/2025
"""
import sys
sys.path.append('.')
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, clean_data
from starter.ml.model import inference, train_model, compute_model_metrics
import pickle
import logging


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_data():
    """ Test data csv """

    data = pd.read_csv('data/census.csv')
    assert data.shape[0] > 0

@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("data/census.csv")
    df = clean_data(df)
    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def model(data):
    X_train, y_train, _, _ = data
    model = train_model(X_train, y_train)
    return model


def test_train_model(data, model):
    X_train, y_train, _, _ = data
    assert model is not None


def test_compute_model_metrics(data, model):
    X_train, y_train, X_test, y_test = data
    predictions = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(data, model):
    X_train, _, _, _ = data
    predictions = inference(model, X_train)
    assert predictions is not None


def test_model():
    """ Test Random Forest model """
    model = pickle.load(open("./model/model.pkl", "rb"))
    assert isinstance(model, RandomForestClassifier)