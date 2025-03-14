import json
import requests

data = {"age": 32,
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
        }

response = requests.post(
    "https://final-nd0821-c3-starter-code-master.onrender.com/predict", data=json.dumps(data))

print(response.status_code)
print(response.json())


