"""
Module for creating slices and computing prediction metrics on them.
"""

import os
from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

# Initialising the logger
logging.basicConfig(filename='./logs.log',
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def data_slicing(data, cat_features):
    """Function for data slicing model performance given certain categorical column"""
    train, test = train_test_split(data, test_size=0.20)

    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    lb = pickle.load(open("./model/lb.pkl", "rb"))
    slice_result = {'feature': [], 'category': [], 'precision': [], 'recall': [], 'Fbeta': []}

    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp, categorical_features=cat_features, label='salary', training=False,
                encoder=encoder, lb=lb
            )

            y_pred = model.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            slice_result['feature'].append(cat)
            slice_result['category'].append(cls)
            slice_result['precision'].append(precision)
            slice_result['recall'].append(recall)
            slice_result['Fbeta'].append(fbeta)

    df = pd.DataFrame.from_dict(slice_result)
    df.to_csv('slice_output.txt', index=False)
    logger.info(f"Slice output saves as slice_output.txt")
