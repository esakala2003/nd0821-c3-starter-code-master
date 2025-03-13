import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"
data = {
    'age': 38,
    'fnlgt': 15,
    'education_num': 1,
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 5
}
#Call each API endpoint and store the responses
response1 = client.post("/predict/", json=data)
#response3 = requests.post(f"{URL}/predict", json="data").text
response2 = requests.get(f"{URL}/").text

print(response1)