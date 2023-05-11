'''
This is a script to use trained models to make predictions.
'''

from pipelines import MyPipeline
from dotenv import load_dotenv
import pandas as pd
import yaml
import os
from src.aspect_classification.data.data_cleaning


# Load environment variables from .env file
load_dotenv()

with open('src/aspect_classification/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    
    
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def prepare_data():
    test_data = pd.read_csv(test_data_path)
    processed_data = 
    my_pipeline.predict()


if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/google_reviews.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/' 
    results_path = os.getenv('LOCAL_ENV') + 'results/aspect_classification/' + params['model_name'] + '.csv'
    