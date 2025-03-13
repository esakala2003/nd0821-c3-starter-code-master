"""
FastAPI interface used to make inference on census data using a neural network. The app is deployed on Heroku.

Author: Emmanuel Sakala
Date: 12/03/2025
"""
import pandas as pd

#from starter.ml.model import train_model, inference
#from starter.ml.data import process_data
#from starter.train_model import train_save_model

from fastapi import FastAPI
import pickle
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel, Field
import os
import logging
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import inference
#import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


class Person(BaseModel):
    """
    Input data object for model inference
    """

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2100,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    # This code is necessary for Heroku to use dvc
    logger.info("Running DVC")
    os.system("dvc config core.no_scm true")
    pull_err = os.system("dvc pull")
    if pull_err != 0:
        exit(f"dvc pull failed, error {pull_err}")
    else:
        logger.info("DVC Pull worked.")
    logger.info('removing dvc files')
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def welcome_message():
    return {"Greetings": "Welcome to the Income Prediction API"}


@app.post("/predict")
async def predict(person: Person):
    logger.info("starting POST request")
    """POST method for model inference"""
    artifact_dict = pickle.load(open("model/model.pkl", "rb"))
    model = artifact_dict["model"]
    encoder = artifact_dict["encoder"]
    lb = artifact_dict["lb"]
    data = pd.DataFrame([person.dict(by_alias=True)])
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
    X_test, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    return {"prediction": preds.item()}
