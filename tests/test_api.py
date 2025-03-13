"""
This script holds the test functions for api module
Author: Emmanuel Sakala
Date: 12/03/2025
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope='module')
def client():
    test_client = TestClient(app)
    return test_client

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

def test_post_above_50k():
    """ Test the output for salary is >50k """

    r = client.post('/prediction', json={
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {'Predicted Income': ' >50K'}

def test_post_below_50k():
    """ Test the output for salary is <50k """

    r = client.post('/prediction', json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {'Predicted Income': ' <=50K'}