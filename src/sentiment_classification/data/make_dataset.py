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
    selected_rows = preprocessor.remove_rows(sample_test_df, 'Gold Sentiment Labels')
    # renamed_sentiments = preprocessor.relabel(selected_rows, ['Gold Sentiment Labels', 'Sentiment'])
    # renamed_sentiments.to_csv(processed_test_path)
    return selected_rows


# Call main function
if __name__ == '__main__':
    test_file_path = os.environ.get('LOCAL_ENV') + '/data/interim/predicted_aspect_labels.csv'
    processed_test_path = os.environ.get('LOCAL_ENV') + '/data/interim/data_for_sentiment_prediction.csv'
    make_testset()
    print('Done !')