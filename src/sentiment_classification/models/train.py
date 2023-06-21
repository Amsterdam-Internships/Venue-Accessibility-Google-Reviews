from pipelines import SentimentPipeline
from sklearn.model_selection import GridSearchCV
import os
from dotenv import load_dotenv
import sys    
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from aspect_classification.data.data_cleaning import bert_processing
import joblib
import pandas as pd

import yaml

# Load environment variables from .env file
config_path = os.getenv('LOCAL_ENV') + 'src/sentiment_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    
    
my_pipeline = SentimentPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def split_data(euans_data):
    euans_reviews = euans_data.Text.values.tolist()
    euans_labels = euans_data.Sentiment.values.tolist()
    return euans_reviews, euans_labels

def train_classic_models():
    grid_search = GridSearchCV(estimator=my_pipeline.get_sk_pipeline(),
    param_grid=my_pipeline.get_params(),
    cv=5, n_jobs=3, verbose=3, scoring='accuracy')
    euans_data = pd.read_csv(loaded_data_path)
    X_train, y_train = split_data(euans_data)
    trained_model = grid_search.fit(X_train, y_train)
    print('training of classic models has finished !')
    save_path = saved_model_path + '/gridsearch.joblib'
    joblib.dump(trained_model, save_path)

def train_bert_models():
    euans_data = pd.read_csv(loaded_data_path)
    X_train, y_train = split_data(euans_data)
    X_train = bert_processing(X_train)
    trained_model = my_pipeline.fit(X_train[:800], y_train[:800])
    print('training of BERT models has finished !')
    save_path = saved_model_path + '/bert.joblib'
    print(save_path)
    joblib.dump(trained_model, save_path)

if __name__ == '__main__':
    # Get the file paths from environment variables
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/processed_euans_reviews.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/sentiment_analysis'
    if params['pipeline_type'] == 'default':
        train_classic_models()
    else:
        train_bert_models()
