# author: GROUP 12
# date: 2021-11-25
	
'''This script first tunes a LogisticRegression model using grid search, and then tests the model
before outputting a confusion matrix and classification report to our results folder. 
	
Usage: analysis.py --train_path=<train_path> --test_path=<test_path> --out_dir=<out_dir>
	
Options:
--train_path=<train_path>   Path to the training data file (must be in standard csv format)
--test_path=<test_path>    Path to the test data file (must be in standard csv format)
--out_dir=<out_dir>    Output directory path where the result figures will be saved
'''

import os
import requests
import pandas as pd
import numpy as np
from docopt import docopt
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV

opt = docopt(__doc__)
class_report_name = "classification_report.csv"
confusion_matrix_name = "confusion_matrix.png"
feature_importance_filename = "feature_importance.csv"

def main(train_path, test_path, out_dir):


    X_train = None
    y_train = None
    X_test = None
    y_test = None

    # Read train & test data
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print("Unable to read training/test data. Please check filepath's.")

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            print("Wasn't able to create output directory. Check permissions.")
    
    # Split into features/targets
    X_train, y_train, X_test, y_test = split_feature_targets(train_df, test_df)

    # Tests for data
    test_empty(X_train)
    test_empty(X_test)
    test_empty(y_train)
    test_empty(y_test)

    test_columns(X_test)
    test_columns(X_test)

    test_columns_y(y_test)
    test_columns_y(y_train)


    # Build Preprocessor
    numeric_features = ['MonthlyCharges', 'tenure', 'TotalCharges']

    categorical_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod', 'SeniorCitizen']

    # Gender and customerID features were dropped in pre_process_script.py
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Build pipeline for hyperparameter optimization
    lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=10000))

    param_grid = {
        "logisticregression__C": 10.0 ** np.arange(-4, 6, 1),
        "logisticregression__class_weight":[None, "balanced"]
    }

    # We're concerned with finding parameters that optimize f1 score
    grid_search = GridSearchCV(
        lr_pipe,
        param_grid,
        cv=4,
        scoring="f1"
    )

    # Get best parameters from grid_search
    best_coefs = grid_search.fit(X_train, y_train).best_params_

    # Define best model pipeline
    best_lr_pipe = make_pipeline(preprocessor, 
                                LogisticRegression(C=best_coefs["logisticregression__C"], 
                                                   class_weight=best_coefs["logisticregression__class_weight"]))

    # Fit best model on training data
    best_lr_pipe.fit(X_train, y_train)

    # Grab learned coefficients from model. We'll need these to report on which features are most 
    # meaningful for predicting churn. 
    feats = best_lr_pipe.named_steps["logisticregression"].coef_[0]

    # Build list of column names. There are some new columns from one hot encoder. 
    cols = (numeric_features + 
        list(preprocessor.named_transformers_["onehotencoder"].get_feature_names_out()))


    # Build & export dataframe of most positively & negatively correlated features
    feature_imp_df = pd.DataFrame(
        data=feats,
        index=cols,
        columns=["Coefficient"],
    )
    
    feature_imp_df.to_csv(os.path.join(out_dir, feature_importance_filename))

    # Run best model on test data
    preds = best_lr_pipe.predict(X_test)

    # Generate classification report
    class_report = classification_report(
        y_test, preds, target_names=["non-Churn", "Churn"], output_dict=True
    )

    class_report_df = pd.DataFrame(class_report).transpose()

    class_report_df.to_csv(os.path.join(out_dir, class_report_name))


    #Testing if results DataFrames are empty
    test_results_empty(class_report_df)
    test_results_empty(feature_imp_df)
    


    # Generate confusion matrix
    cm = ConfusionMatrixDisplay.from_estimator(
        best_lr_pipe, X_test, y_test, values_format="d", display_labels=["Non Churn", "Churn"]
    )
    cm.figure_.savefig(os.path.join(out_dir, confusion_matrix_name))
    
def split_feature_targets(train_df, test_df):
    """
    Splits the training dataframe and test dataframe into X_train, y_train & X_test, y_test

    Parameters:
    train_df (pandas DataFrame object): the training dataframe
    test_df (pandas DataFrame object): the test dataframe

    Returns:
    X_train (pandas DataFrame object): the training dataframe. Just features and no target. 
    y_train (pandas DataFrame object): the training target values. 
    X_test (pandas DataFrame object): the test dataframe. Just features and no target
    y_test (pandas DataFrame object): the test target values. 
    """
    X_train = train_df.drop(columns=["Churn"])
    X_test = test_df.drop(columns=["Churn"])

    y_train = train_df["Churn"]
    y_test = test_df["Churn"]

    return X_train, y_train, X_test, y_test

def build_preprocessor(numeric_features, categorical_features):

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("simpleimputer", SimpleImputer()),
        ("standardscaler", StandardScaler() )
        ]
    )

    # Build preprocessor
    preprocessor = make_column_transformer(
        (numeric_pipeline, numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=True), categorical_features)
    )

    return preprocessor


# Testing Functions
def test_empty(data):
    
    assert data.empty == False, "Data file path/URL is incorrect"

def test_columns(data):
    
    assert data.columns[0] != 'customerID', "Data columns are incorrect, customerID should be dropped"
    assert data.columns[0] != 'gender', "Data columns are incorrect, gender should be dropped"
    assert data.columns[0] == 'SeniorCitizen', "Data columns are incorrect"
    assert data.columns[1] == 'Partner', "Data columns are incorrect"

def test_columns_y(data):
    
    assert data.name == 'Churn', "Target column is incorrect" 

def test_results_empty(data):
    
    assert data.empty == False, "Results DataFrame is Empty" 

if __name__ == "__main__":
    
    # Have the user input train data path, test data path, and output directory
    main(opt["--train_path"], opt["--test_path"], opt["--out_dir"])