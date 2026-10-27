# author: GROUP 12
# date: 2021-11-19
	
'''This script downloads a data file in csv format. 
This script takes an unquoted data file path to a csv file, 
the name of the file type to write the file to (ex. csv), 
and the name of a file path to write locally (including the name of the file).
	
Usage: data_download.py --file_path=<file_path> --out_type=<out_type> --out_file=<out_file>
	
Options:
--file_path=<file_path>   Path to the data file (must be in standard csv format)
--out_type=<out_type>    Type of file to write locally (script supports either feather or csv)
--out_file=<out_file>    Path (including filename) of where to locally write the file
'''
	
import pandas as pd
import numpy as np
from docopt import docopt
import os
import requests
	
opt = docopt(__doc__)
	
def main(file_path, out_type, out_file):

    try: 
        request = requests.get(file_path)
        request.status_code == 200
    except Exception as req:
        print("Website at the provided url does not exist.")
        print(req)

    # read in data and test it
    data = None
    data = pd.read_csv(file_path)


    # Tests for raw data
    test_path(data)
    test_columns(data)

    # Create new file path if it doesn't exist
    if out_type == "csv":
        try:
            data.to_csv(out_file, index = False)
        except:
            os.makedirs(os.path.dirname(out_file))
            data.to_csv(out_file, index = False)

  
def test_path(data):

    assert data.empty == False, "Data file path/URL is incorrect"

def test_columns(data):
    
    assert data.columns[0] == 'customerID', "Data columns are incorrect"
    assert data.columns[1] == 'gender', "Data columns are incorrect"
    assert data.columns[2] == 'SeniorCitizen', "Data columns are incorrect"
    assert data.columns[3] == 'Partner', "Data columns are incorrect"

    assert data.columns[20] == 'Churn', "Target column is incorrect" 
    

  
if __name__ == "__main__":
    
    # Call main method, and have the user input file path, out type, and out path
    main(opt["--file_path"], opt["--out_type"], opt["--out_file"])
