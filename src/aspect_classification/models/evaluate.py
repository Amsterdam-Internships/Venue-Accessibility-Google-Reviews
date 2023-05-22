'''
This is a script to use trained models to make predictions.
'''
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from aspect_classification.data.data_cleaning import bert_processing
from transformers import AutoTokenizer
from pipelines import MyPipeline
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import yaml
import os

# Load environment variables from .env file
load_dotenv()
config_path = os.getenv('LOCAL_ENV') + 'src/aspect_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
     
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])


def remove_rows(test_data):
    # Remove the redundant rows
    test_data = test_data.dropna(subset=['Improved Aspect Label'])
    test_data['Aspect Label'] = test_data['Improved Aspect Label'].str.split(' & ')
    return test_data


def generate_results():
    test_data = pd.read_csv(test_data_path)
    print(test_data.head())
    #selected_rows = remove_rows(test_data)
    # Select the input features
    google_reviews = test_data['Review Text'].values.tolist()
    # Select target labels
    y_true = test_data['Aspect'].values.tolist()
    # Process reviews
    processed_reviews = bert_processing(google_reviews)
    y_pred = my_pipeline.predict(processed_reviews)
    eval_metrics = my_pipeline.evaluate(y_true, y_pred)
    save_results(eval_metrics, y_pred)

def save_results(eval_metrics, y_pred):
    # Convert y_pred to a pandas DataFrame
    predicted_labels_df = pd.DataFrame({'Predicted Aspect Labels': y_pred}, index=range(len(y_pred)))

    # Save the predicted labels as a CSV file
    predicted_labels_path = interim_path + "/predicted_aspect_labels.csv"
    predicted_labels_df.to_csv(predicted_labels_path, index=False)
    
    # # Save the classification report as a text file
    report_path = results_path+"_classification_report.tex"
    pd.DataFrame(eval_metrics).to_csv(results_path+'.csv')
    report_df = pd.DataFrame(eval_metrics).transpose()
    report_df.to_latex(report_path)


if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/test_example.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/transformer_models/bert.joblib'
    # Define the directory path
    names = params['model_name'].split("/")
    results_path = os.getenv('LOCAL_ENV') + f"results/aspect_classification/{names[1]}"
    interim_path = os.getenv('LOCAL_ENV') + 'data/interim'

    # Call the function to generate results
    generate_results()
