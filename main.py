"""
FastAPI interface used to make inference on census data using a neural network. The app is deployed on Heroku.

Author: Emmanuel Sakala
Date: 12/03/2025
"""
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference
from fastapi import FastAPI
import pickle
from pydantic import BaseModel, Field
import os
import logging
import uvicorn


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

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


class InputData(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


# get encoder, trained model
model = pickle.load(open("./model/model.pkl", "rb"))
encoder = pickle.load(open("./model/encoder.pkl", "rb"))
lb = pickle.load(open("./model/lb.pkl", "rb"))

app = FastAPI()


@app.get("/")
async def welcome_message():
    return {'message': '"Welcome to the Income Prediction API'}


@app.post("/predict")
async def predict(input_data: InputData):
    logger.info("starting POST request")
    """POST method for model inference"""

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
    sample = {key.replace('_', '-'): [value] for key, value in input_data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(sample)
    input_data = input_data.dropna()
    x, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=x)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    return {"Predicted Income": str_out}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=5000, reload=True)
