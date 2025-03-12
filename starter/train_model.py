# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data, clean_data
from ml.model import train_model, inference, compute_model_metrics
from slice_dataset import data_slicing
import pandas as pd
import os
import pickle

# Add the necessary imports for the starter code.
def train_save_model():
    # Add code to load in the data.
    #dirname = os.path.dirname(__file__)
    #data = pd.read_csv(os.path.join(dirname, "../data/census.csv"))
    data = pd.read_csv("./data/census.csv")
    data = clean_data(data)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Train and save a model.
    trained_ml = train_model(X_train, y_train)
    #pickle.dump(trained_ml, open(os.path.join(dirname, "../model/model.pkl"), 'wb'))
    #pickle.dump(encoder, open(os.path.join(dirname, "../model/encoder.pkl"), 'wb'))

    pickle.dump(trained_ml, open("./model/model.pkl", 'wb'))
    pickle.dump(encoder, open("./model/encoder.pkl", 'wb'))

    # Data slicing function on certain column (using education as example)
    data_slicing(test, cat_features, trained_ml, encoder, lb, "education")

    # Show the overall performance for writing model card
    predictions = inference(trained_ml, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    print(f"Overall Performance: precision:{precision}, recall:{recall}, fbeta:{fbeta}")
    return precision, recall, fbeta

