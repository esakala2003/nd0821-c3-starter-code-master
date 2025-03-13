"""
This script holds the test functions for api module
Author: Emmanuel Sakala
Date: 12/03/2025
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
import json
import pandas as pd
from main import app

@pytest.fixture(scope='module')
def client():
    mock_client = TestClient(app)
    return mock_client

def test_greetings():
    """
    Tests GET greetings function
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"
    assert response.json() == 'Welcome to the Income Prediction API'

@pytest.fixture(scope='module')
def test_predict_status():
    """
    Tests POST predict function status
    """
    data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"

def test_api_post_positive(positive_example, client):
    """
    Test that the prediction we get from using the api is identical to the one we get from inferring
    directly from the model. We consider both an example where the model predicts a positive outcome and a case
    where the model predicts a negative outcome.
    """
    type_ex, example = 'positive', positive_example
    response = client.post("/predict", json=positive_example)
    output = response.json()['predicted_salary_class']
    data = pd.DataFrame([example])
    cat_features = get_cat_features()
    model = get_trained_mlp()
    x, _, _, _, _ = process_data(data, categorical_features=cat_features, label="salary",
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = inference(model, x)
    expected_output = predicted[0]
    assert response.status_code == 200
    assert output == expected_output, f"API prediction failed for an example labelled as {type_ex} by the model"


def test_api_post_negative(predict_request, negative_example, client):
    """
    Test that the prediction we get from using the api is identical to the one we get from inferring
    directly from the model. We consider both an example where the model predicts a positive outcome and a case
    where the model predicts a negative outcome.
    """
    type_ex, example = 'negative', negative_example
    response = client.post("/predict", data=json.dumps(negative_example))
    output = response.json()['predicted_salary_class']
    data = pd.DataFrame([example])
    cat_features = get_cat_features()
    model = get_trained_mlp()
    x, _, _, _, _ = process_data(data, categorical_features=cat_features, label="salary",
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = inference(model, x)
    expected_output = predicted[0]
    assert response.status_code == 200
    assert output == expected_output, f"API prediction failed for an example labelled as {type_ex} by the model"
