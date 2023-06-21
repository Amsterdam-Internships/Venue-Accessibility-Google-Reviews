'''
This is a script to use trained models to make predictions.
'''
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from aspect_classification.data.data_cleaning import bert_processing
from aspect_classification.data.preprocessing import Preprocessor
from transformers import AutoTokenizer
from pipelines import MyPipeline
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import yaml
import os

# Load environment variables from .env file
load_dotenv()
config_path = os.getenv('LOCAL_ENV') + 'src/aspect_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
     
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])
preprocessor = Preprocessor()

def generate_results(test_data):
    # Select annotated data only for the aspect classification task
    annotated_data = preprocessor.remove_rows(test_data)
    # Select the input features
    google_reviews = annotated_data['Sentences'].values.tolist()
    # Select target labels
    gold_labels = annotated_data['Aspect'].values.tolist()
    # Make predictions on reviews
    processed_reviews = bert_processing(google_reviews)
    
    predicted_labels = my_pipeline.predict(processed_reviews)
    # Assuming annotated_data is your DataFrame
    annotated_data['Predicted Aspect Labels'] = pd.Series(predicted_labels)
    # get and save metrics
    metrics = my_pipeline.evaluate(gold_labels, annotated_data['Predicted Aspect Labels'])
    save_results(metrics, annotated_data)
    
def save_results(eval_metrics, predicted_df):
    # Save the predicted labels as a CSV file
    predicted_labels_path = interim_path + "/predicted_aspect_labels.csv"
    predicted_df.to_csv(predicted_labels_path)
    
    # Create a DataFrame from the eval_metrics dictionary
    eval_metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index').transpose()
    
    # Save the evaluation metrics as a CSV file
    results_path_csv = results_path + '.csv'
    eval_metrics_df.to_csv(results_path_csv, index=False)
    
    # Save the classification report as a text file
    report_path = results_path + "_classification_report.tex"
    print(eval_metrics_df.columns)
    eval_metrics_df.to_latex(report_path, index=False)

if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/processed_google_sample_reviews.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/transformer_models/bert.joblib'
    # Define the directory path
    names = params['model_name'].split("/")
    results_path = os.getenv('LOCAL_ENV') + f"results/aspect_classification/{names[1]}"
    interim_path = os.getenv('LOCAL_ENV') + 'data/interim'

    # Call the function to generate results
    test_data = pd.read_csv(test_data_path)
    generate_results(test_data)
