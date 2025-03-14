"""
This script holds the test functions for api module
Author: Emmanuel Sakala
Date: 12/03/2025
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
import os
import sys

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from main import app

client = TestClient(app)

def test_greetings():
    """
    Tests GET greetings function
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': '"Welcome to the Income Prediction API'}

def test_post_above_50k():
    """ Test the output for salary is >50k """

    response = client.post('/predict', json={
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert response.status_code == 200
    assert response.json() == {'Predicted Income': '>50K'}

def test_post_below_50k():
    """ Test the output for salary is <50k """

    response = client.post('/predict', json={
        "age": 32,
        "workclass": "Private",
        "fnlgt": 27882,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Other-relative",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 2205,
        "hours_per_week": 40,
        "native_country": "Holand-Netherlands"
    })

    assert response.status_code == 200
    assert response.json() == {'Predicted Income': '<=50K'}