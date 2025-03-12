"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the test functions for api module
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from app.api import app

@pytest.fixture(scope='module')
def client():
    mock_client = TestClient(app)
    return mock_client

def test_api_get_root(client):
    r = client.get("/")
    assert r.status_code == 200
    output = r.json()
    expected_output = {'greeting': 'Welcome! This API predicts income category using Census data.'}
    assert output == expected_output
