'''
This is a script to load, clean and save data.

'''

import pandas as pd 
import os
from dotenv import load_dotenv
from preprocessing import Preprocessor

# Load environment variables from .env file
load_dotenv()
preprocessor = Preprocessor()

    
def make_testset():
    sample_test_df = pd.read_csv(test_file_path)
    print(sample_test_df.columns)
    renamed_sentiments = preprocessor.relabel(sample_test_df, 'Gold Sentiment Labels')
    selected_rows = preprocessor.remove_rows(renamed_sentiments, 'Sentiment')
    selected_rows.to_csv(processed_test_path)


# Call main function
if __name__ == '__main__':
    test_file_path = os.environ.get('LOCAL_ENV') + '/data/interim/predicted_aspect_labels.csv'
    processed_test_path = os.environ.get('LOCAL_ENV') + '/data/interim/data_for_sentiment_prediction.csv'
    make_testset()
    print('Done !')