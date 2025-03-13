import os
import pandas as pd
import subprocess
from starter.ml.data import clean_data, process_data
from starter.train_model import train_save_model
from sklearn.model_selection import train_test_split

#print(os.listdir())
data = pd.read_csv("./data/census.csv")
#result = subprocess.run(["python", "./starter/ml/data.py"])
result = clean_data(data)
train, test = train_test_split(result, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

train_save_model()