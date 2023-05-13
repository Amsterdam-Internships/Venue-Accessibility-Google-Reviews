'''
This is a script to use trained models to make predictions.
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from aspect_classification.data.data_cleaning import bert_processing
from pipelines import MyPipeline
from dotenv import load_dotenv
import pandas as pd
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
    confusion = confusion_matrix(y_true, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': confusion}

def generate_results():
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.dropna(subset=['Aspect Label'])
    google_reviews = test_data['Aspect Label'].values.tolist()
    processed_data = bert_processing(google_reviews)
    y_pred = my_pipeline.predict(processed_data[:842])
    print(google_reviews)
    eval_report = evaluate(google_reviews, y_pred)
    df = pd.DataFrame(eval_report)
    df.to_csv(results_path, index=False)
    


if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/trial labels.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/transformer_models/bert.joblib' 
    results_path = os.getenv('LOCAL_ENV') + 'results/aspect_classification/' + params['model_name'] + '.csv'
    generate_results()