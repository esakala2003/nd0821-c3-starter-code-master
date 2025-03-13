"""
Script to train machine learning model.
Authur: Emmanuel Sakala
Date: 13/03/2025
"""

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, clean_data
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.slice_dataset import data_slicing
import pandas as pd
import pickle
import logging

# Initialising the logger
logging.basicConfig(filename='./logs.log',
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the necessary imports for the starter code.
def train_save_model():
    # Add code to load in the data.
    """
    Function to train and save the trained model
    Returns
    -------
    Trained ML model in .pkl format
    Training scores: precision, recall, fbeta
    """
    logging.info("Training the ML......")
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
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb,
    )
    # Train and save a model.
    trained_ml = train_model(X_train, y_train)
    logger.info('Saving the trained model....')
    pickle.dump(trained_ml, open("./model/model.pkl", 'wb'))
    pickle.dump(encoder, open("./model/encoder.pkl", 'wb'))
    pickle.dump(lb, open("./model/lb.pkl", 'wb'))

    # Data slicing function on certain column (using education as example)
    data_slicing(test, cat_features)

    # Show the overall performance for writing model card
    predictions = inference(trained_ml, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    print(f"Overall Performance: precision:{precision}, recall:{recall}, fbeta:{fbeta}")
    logger.info(f"Overall Performance: precision: {precision: 2f}, recall: {recall: 2f}, fbeta: {fbeta: 2f}")
    logger.info(f"Trained model files saved in ./model folder")
    return precision, recall, fbeta

if __name__ == "__main__":
    train_save_model()