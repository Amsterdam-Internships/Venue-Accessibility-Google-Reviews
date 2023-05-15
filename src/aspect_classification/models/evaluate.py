'''
This is a script to use trained models to make predictions.
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from aspect_classification.data.data_cleaning import bert_processing
from sklearn.preprocessing import LabelEncoder
from pipelines import MyPipeline
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import yaml
import os

# Load environment variables from .env file
load_dotenv()

with open('src/aspect_classification/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    

    
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = np.array2string(confusion_matrix(y_true, y_pred))
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': confusion}

def remove_rows(test_data):
    # Remove the redundant rows
    test_data = test_data.dropna(subset=['Aspect Label'])
    test_data = test_data[test_data['Aspect Label'] != 'Nonsense']
    return test_data

def encode_input(data):
    encoder = LabelEncoder()
    y_true = np.array(encoder.fit_transform(data))
    return y_true
    
def generate_results():
    test_data = pd.read_excel(test_data_path)
    selected_rows = remove_rows(test_data)
    # Select the input features
    google_reviews = selected_rows['Text'].values.tolist()
    # Select target labels
    google_labels = selected_rows['Aspect Label'].values.tolist()
    # Process reviews
    processed_reviews = bert_processing(google_reviews)
    y_pred = my_pipeline.predict(processed_reviews)
    y_true = encode_input(google_labels)

    # Convert y_true and y_pred to lists of strings
    y_true = [str(label) for label in y_true]
    y_pred = [str(label) for label in y_pred]

    eval_report = evaluate(y_true, y_pred)

    df = pd.DataFrame([eval_report])
    df.to_csv(results_path, index=False)

if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/revised_trial_labels.xlsx'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/transformer_models/bert.joblib'
    # Define the directory path
    names = params['model_name'].split("/")
    results_path = os.getenv('LOCAL_ENV') + f"results/aspect_classification/{names[1]}.csv"

    # Call the function to generate results
    generate_results()
