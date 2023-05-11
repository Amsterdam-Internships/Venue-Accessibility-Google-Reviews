from pipelines import MyPipeline
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
import joblib
import pandas as pd
import os
import yaml

# Load environment variables from .env file
load_dotenv()
with open('src/aspect_classification/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    
    
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def split_data(euans_data):
    euans_reviews = euans_data.Text.values.tolist()
    euans_labels = euans_data.Aspect.values.tolist()
    return euans_reviews, euans_labels

def train_classic_models():
    grid_search = GridSearchCV(estimator=my_pipeline.get_sk_pipeline(),
    param_grid=my_pipeline.get_params(),
    cv=5, n_jobs=3, verbose=3, scoring='accuracy')
    euans_data = pd.read_csv(loaded_data_path)
    X_train, y_train = split_data(euans_data)
    trained_model = grid_search.fit(X_train[:10000], y_train[:10000])
    print('training of classic models has finished !')
    save_path = saved_model_path + '/gridsearch.joblib'
    joblib.dump(trained_model, save_path)

def train_bert_models():
    euans_data = pd.read_csv(loaded_data_path)
    X_train, y_train = split_data(euans_data)
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    trained_model = my_pipeline.fit(X_train, y_train)
    print('training of BERT models has finished !')
    save_path = saved_model_path + '/bert.joblib'
    print(save_path)
    joblib.dump(trained_model, save_path)

if __name__ == '__main__':
    # Get the file paths from environment variables
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/processed/aspect_classification_data/euans_reviews.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification'
    
    train_classic_models()
    train_bert_models()
