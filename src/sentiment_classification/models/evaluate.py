'''
This is a script to use trained models to make predictions.
'''
import sys
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from transformers import TrainingArguments, pipeline
from sentiment_pipeline import SentimentClassificationPipeline, MultiClassTrainer
from sentiment_classification.data.preprocessing import Preprocessor
import pandas as pd
import yaml

preprocessor = Preprocessor()

config_path = os.getenv('LOCAL_ENV') + '/src/sentiment_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params['bert_params']
     
my_pipeline = SentimentClassificationPipeline(pipeline_type='transformer', model_type=params['model_name_or_path'])


def generate_results(test_data):
    # Select annotated data only for the aspect classification task
    eval_dataset = preprocessor.create_datasets(test_data[['Sentences', 'Sentiment']])    
    # Define the TrainingArguments for evaluation
    eval_args = TrainingArguments(
        output_dir="./results/sentiment_classification",
        evaluation_strategy="epoch",
        do_predict=True,
        num_train_epochs=0,  # Set to 0 to only perform evaluation
    )
    
    # Create a dummy Trainer for evaluation
    trainer = MultiClassTrainer(
        model=pipeline('text-classification', model=loaded_model_path, tokenizer=loaded_model_path),  # Your loaded model
        args=eval_args,
        eval_dataset=eval_dataset,
        compute_metrics=my_pipeline.compute_metrics,
    )
    
    # Perform evaluation
    evaluation_result = trainer.evaluate(eval_dataset=eval_dataset)
    # Assuming annotated_data is your DataFrame
    test_data['Predicted Sentiment Labels'] = my_pipeline.decode_labels(evaluation_result['predictions'])
    save_results(evaluation_result, test_data)


def save_results(eval_metrics, predicted_df):
    # Save the predicted labels as a CSV file
    predicted_labels_path = interim_path + "/predicted_sentiment_labels.csv"
    predicted_df.to_csv(predicted_labels_path)
    
    # Create a DataFrame from the eval_metrics dictionary
    eval_metrics_df = pd.DataFrame.from_dict(eval_metrics, orient='index').transpose()
    
    # Save the evaluation metrics as a CSV file
    results_path_csv = results_path + '.csv'
    eval_metrics_df.to_csv(results_path_csv, index=False)
    
    # Save the classification report as a text file
    report_path = results_path + "_senitment_classification_report.tex"
    eval_metrics_df.to_latex(report_path, index=False)


if __name__ == '__main__':
    # Define the directory path
    names = params['model_name_or_path'].split("/")[-1] if "/" in params['model_name_or_path'] else params['model_name_or_path']
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + '/data/interim/predicted_aspect_labels.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + f'/models/sentiment_classification/{names}'
    results_path = os.getenv('LOCAL_ENV') + f"/results/sentiment_classification/{names}"
    interim_path = os.getenv('LOCAL_ENV') + '/data/interim'

    # Call the function to generate results
    test_data = pd.read_csv(test_data_path)
    generate_results(test_data)
