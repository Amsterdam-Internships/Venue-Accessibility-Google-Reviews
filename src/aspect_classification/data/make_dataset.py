'''
This is a script to load, clean and save data.

'''

import pandas as pd 
import os
from data_cleaning import cleaning_selector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def make_trainset():
    training_data = pd.read_excel(training_file_path)
    clean_train_df = cleaning_selector(training_data, ["Aspect", "Rating", "Review", "Venue"])
    clean_train_df.to_csv(processed_train_path)
    
def make_testset():
    test_data = load_data()
    clean_test_df = cleaning_selector(test_data, ["Name","Review Rate", "Review Text"])
    clean_test_df.to_csv(processed_test_path)

def load_data():
    df_list = []
    for filename in os.listdir(test_file_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(test_file_path, filename))
            df_list.append(df)
    google_data = pd.concat(df_list, axis=0, ignore_index=True)
    return google_data


# Call main function
if __name__ == '__main__':
    training_file_path = os.environ.get('LOCAL_ENV') + 'data/raw/train/EuansGuideData.xlsx'
    test_file_path = os.environ.get('LOCAL_ENV') + 'data/raw/test/GoogleReviews'
    processed_train_path = os.environ.get('LOCAL_ENV') + 'data/processed/aspect_classification_data/processed_euans_reviews.csv'
    processed_test_path = os.environ.get('LOCAL_ENV') + 'data/processed/aspect_classification_data/processed_google_reviews.csv'
    make_trainset()
    make_testset()
    print('Done !')