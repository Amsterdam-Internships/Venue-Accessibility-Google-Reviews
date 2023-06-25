'''
This is a script to use trained models to make predictions.
'''
import sys
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from aspect_classification.data.data_cleaning import bert_processing
from pipelines import SentimentPipeline
from sentiment_classification.data.preprocessing import Preprocessor
import pandas as pd
import yaml


preprocessor = Preprocessor()

config_path = os.getenv('LOCAL_ENV') + '/src/sentiment_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
     
my_pipeline = SentimentPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def select_rows(test_data):
    test_data = test_data.drop(test_data[test_data['Sentiment'] == 'Neutral'].index)
    return test_data

def generate_results(test_data):
    annotated_data = select_rows(test_data)
    # Select the input features
    google_reviews = annotated_data['Sentences'].values.tolist()
    # Select target labels
    gold_labels = annotated_data['Sentiment'].values.tolist()
    # Make predictions on reviews
    processed_reviews = bert_processing(google_reviews)
    predicted_labels = my_pipeline.predict(processed_reviews)
    # Assuming annotated_data is your DataFrame
    annotated_data['Predicted Sentiment Labels'] = predicted_labels
    # get and save metrics
    metrics = my_pipeline.evaluate(gold_labels, annotated_data['Predicted Sentiment Labels'])
    save_results(metrics, annotated_data)


def save_results(eval_metrics, predicted_df):
    # Save the predicted labels as a CSV file
    predicted_labels_path = interim_path + "/predicted_sentiment_labels.csv"
    predicted_df.to_csv(predicted_labels_path, index=False)
    
    # Create a DataFrame from the eval_metrics dictionary
    eval_metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index').transpose()
    
    # Save the evaluation metrics as a CSV file
    results_path_csv = results_path + '.csv'
    eval_metrics_df.to_csv(results_path_csv, index=False)
    
    # Save the classification report as a text file
    report_path = results_path + "senitment_classification_report.tex"
    eval_metrics_df.to_latex(report_path, index=False)





if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + '/data/interim/predicted_aspect_labels.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + '/models/sentiment_classification/transformer_models/bert.joblib'
    # Define the directory path
    names = params['model_name'].split("/")
    results_path = os.getenv('LOCAL_ENV') + f"/results/sentiment_classification/{names[1]}"
    interim_path = os.getenv('LOCAL_ENV') + '/data/interim'

    # Call the function to generate results
    test_data = pd.read_csv(test_data_path)
    generate_results(test_data)
