
# author: Adam Morphy
# date: 2021-11-23
	
'''
Cleans, splits and pre-processes the Telco Churn data (from hhttps://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
Writes the training and test data to separate csv files.

Usage: src/pre_process_script.py --input=<input> --out_dir=<out_dir>
  
Options:
--input=<input>       Path (including filename) to raw data (csv file)
--out_dir=<out_dir>   Path to directory where the processed data should be written
'''
	
import pandas as pd
import numpy as np
from docopt import docopt
import os
import requests
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
	
opt = docopt(__doc__)
	
def main(input, out_dir):

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            print("Wasn't able to create output directory. Check permissions.")
            
    # read data and convert class to pandas df
    raw_data = pd.read_csv(input) 

    # Converting TotalCharges from object to float (pd functions astype(float))
    index = 0
    s = raw_data['TotalCharges']

    for c in s:

        if c == ' ':
            raw_data['TotalCharges'][index] = None
    
        index +=1

    raw_data['TotalCharges'] = raw_data['TotalCharges'].astype(float)  

    # Change target variable to a boolean
    raw_data['Churn'] = raw_data['Churn'].replace("Yes", True)
    raw_data['Churn'] = raw_data['Churn'].replace("No", False)

    # Change target variable to a boolean
    raw_data['SeniorCitizen'] = raw_data['SeniorCitizen'].replace(1, "Yes")
    raw_data['SeniorCitizen'] = raw_data['SeniorCitizen'].replace(0, "No")

    # Drop Gender (ethical constraints), and customerID
    raw_data = raw_data.drop(columns=["gender", "customerID"])

    # split into training and test data sets
    train_df, test_df = train_test_split(raw_data, test_size=0.3, random_state=1)

    # Test the new dataframe

    test_empty(train_df)
    test_empty(test_df)

    test_columns(train_df)
    test_columns(test_df)
    
  
    # write training and test data to csv files
    train_df.to_csv((out_dir + 'train_df.csv'), index = False)
    test_df.to_csv((out_dir + 'test_df.csv'), index = False)

    print("Data successfully stored in: ", (out_dir + 'train_df.csv'), " and ", (out_dir + 'train_df.csv'))


# Testing Functions
def test_empty(data):
    
    assert data.empty == False, "Data file path/URL is incorrect"

def test_columns(data):
    
    assert data.columns[0] != 'customerID', "Data columns are incorrect, customerID should be dropped"
    assert data.columns[0] != 'gender', "Data columns are incorrect, gender should be dropped"
    assert data.columns[0] == 'SeniorCitizen', "Data columns are incorrect"
    assert data.columns[1] == 'Partner', "Data columns are incorrect"

if __name__ == "__main__":
    
    # Call main method, and have the user input the raw file, out dir
    main(opt["--input"], opt["--out_dir"])