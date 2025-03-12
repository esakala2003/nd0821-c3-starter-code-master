import os
from ml.data import process_data
from ml.model import inference, compute_model_metrics


def data_slicing(test, cat_features, trained_model, encoder, lb, col):
    """Function for data slicing model performance given certain categorical column"""

    # get distinct column category value
    unique_values = test[col].unique()

    # iterate each value and record the metrics
    for val in unique_values:
        # Fix the feature
        idx = test[col] == val
        temp_test = test[idx]

        # Process this subset of data for testing
        X_test, y_test, encoder, lb = process_data(
            temp_test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Do the inference and Compute the metrics
        predictions = inference(trained_model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, predictions)

        # output the result to slice_output.txt
#        dirname = os.path.dirname(__file__)
#        with open(os.path.join(dirname, "../screenshots/slice_output.txt"), "w") as f:
        with open("./screenshots/slice_output.txt", "w") as f:
            f.write(f"{col}\n")
            for value in unique_values:
                f.write(f"\t {value.strip()}\n")
                f.write(f"\t\t precision:{precision} recall:{recall} fbeta:{fbeta}\n")