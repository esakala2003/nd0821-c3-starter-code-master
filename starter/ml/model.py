"""
This script is for model training and evaluation.
Author: Emmanuel Sakala
Date: 11/03/2025
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import logging

# Initialising the logger
logging.basicConfig(filename='./logs.log',
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    logging.info("Computing model metrics....")
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logging.info("Running model inference.....")
    predictions = model.predict(X)
    return predictions

