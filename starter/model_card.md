# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Emmanuel Sakala created the model. It is a Random Forest model that uses the default hyperparameters in scikit-learn.

## Intended Use
* This model should be used to predict whether income of a US citizen exceeds $50K/yr based on census data.

## Training Data
* The data was obtained from the Census Bureau. The target class is the 'salary', it is a binary variable which value is in '<=50k' and '>50k'.
* The raw data need to be processed by removing spaces.
* The original data set has 32561 rows and 15 columns. An 80-20 split was used to break this into a train and test set. No stratification was done. Within the train set another 80-20 split is done to split data between training and validation so that properly evaluate the performance of hyper-parameters and choose the optimal level for these parameters. Once the hyper-parameters have been set we use all the training data for calibrating the model.
* To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels. A scaler was also used on continuous variables.

## Evaluation Data
* A 20% of the original dataset was used for evalution. The same Enconder and Binarizer used for training was applied to the evaluation set.

## Metrics
* The model was evaluated using precision, recall and F-Beta Score. The values are as follows: Precision: 0.74, Recall: 0.64, F-Beta:0.69.

## Ethical Considerations
* There are no sensitive infomation.
* Data is open sourced on UCI machine learning repository for educational purposes.

## Caveats and Recommendations
* The data was collected in 1996 which does not reflect insights from the modern world. Recent data is recommended to to get a better reflection of the current suitation.
* More work should be done on hyper-parameter tuning to improve the performance of the model as the project focused more on the architecture than accurracy of the model.
