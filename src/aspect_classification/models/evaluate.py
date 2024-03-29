'''
This is a script to use trained models to make predictions.
'''
from dotenv import load_dotenv
load_dotenv(override=True)
import sys
import os
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from aspect_classification.data.preprocessing import Preprocessor
from transformers import TrainingArguments, AutoModelForSequenceClassification
from aspect_pipeline import AspectClassificationPipeline, MultiLabelClassTrainer
import pandas as pd
import yaml

config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params['bert_params']
     
my_pipeline = AspectClassificationPipeline(pipeline_type='transformer', model_type=params['model_name_or_path'])
preprocessor = Preprocessor()

def generate_results(test_data):
    # Select annotated data only for the aspect classification task
    eval_set = preprocessor.create_datasets(test_data[['Sentences', 'Aspect']])
    # Define the TrainingArguments for evaluation
    eval_args = TrainingArguments(
        output_dir="./results/aspect_classification",
        evaluation_strategy="epoch",
        do_predict=True,
        num_train_epochs=0,  # Set to 0 to only perform evaluation
    )
    
    # Create a dummy Trainer for evaluation
    trainer = MultiLabelClassTrainer(
        model=AutoModelForSequenceClassification.from_pretrained(loaded_model_path),  # Your loaded model
        args=eval_args,
        eval_dataset=eval_set,
        compute_metrics=my_pipeline.compute_metrics,
    )
    
    # Perform evaluation
    try:
        evaluation_result = trainer.evaluate(eval_dataset=eval_set)
        # Assuming annotated_data is your DataFrame
        test_data['Predicted Aspect Labels'] = my_pipeline.extract_labels()
        save_results(evaluation_result, test_data)
    except AttributeError as e:
        print(f"{e.name=}") 
    
    
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
    eval_metrics_df.to_latex(report_path, index=False)

if __name__ == '__main__':
    # Define the directory path
    names = my_pipeline.model_name.split('/')[-1] if '/' in my_pipeline.model_name else my_pipeline.model_name
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + '/data/processed/aspect_classification_data/processed_google_sample_reviews.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + f'/models/aspect_classification/transformer_models/{names}'
    results_path = os.getenv('LOCAL_ENV') + f"/results/aspect_classification/{names}"
    interim_path = os.getenv('LOCAL_ENV') + '/data/interim'

    # Call the function to generate results
    test_data = pd.read_csv(test_data_path)
    generate_results(test_data)
