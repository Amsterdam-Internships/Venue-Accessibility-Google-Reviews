'''
This is a script to download or generate data.
'''

import pandas as pd 
import os
import pathlib
import glob
import sys
import nltk
import re
import string 
from src import data_cleaning as dc

# Get current directory
current_dir = os.getcwd()
# Get parent directory
parent_dir = os.path.join(current_dir, '..')
# Append parent directory to sys.path
sys.path.append(parent_dir)


cwd = pathlib.Path.cwd().parent
training_file_path = cwd.joinpath("datasets/EuansGuideData.xlsx")
test_file_path = cwd.joinpath("datasets/GoogleReviews")


all_file_names = glob.glob(str(test_file_path) + "/*.csv")
google_df = [pd.read_csv(file_name, index_col=None, header=0) for file_name in all_file_names]
test_data = pd.concat(google_df, axis=0, ignore_index=True)

training_data = pd.read_excel(training_file_path)
clean_train_df = dc.cleaning_selector(training_data, ["Aspect", "Rating", "Review", "Venue"])
clean_test_df = dc.cleaning_selector(test_data, ["Name","Review Rate", "Review Text"])

clean_train_df.to_csv('datasets/processed data/clean_euans.csv')
clean_test_df.to_csv('datasets/processed data/clean_google.csv')


def load_data():
    pass

def clean_data():
    pass

def save_data():
    pass