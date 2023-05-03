'''
This is a script to train the models.
'''
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from dotenv import load_dotenv
import joblib

# Load environment variables from .env file
load_dotenv()


pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', VotingClassifier([
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression())
        ]))
    ])

parameters = {
    'vectorizer__max_df': (0.5, 0.75, 1.0),
    'vectorizer__ngram_range': ((1, 1), (1, 2)),
    'clf__voting': ('soft', 'hard'),
    'clf__nb__alpha': (0.5, 1),
    'clf__lr__C': (0.1, 1, 10),
}

def split_data(euans_data):
    euans_reviews = euans_data.Text.values.tolist()
    euans_labels = euans_data.Aspect.values.tolist()
    print(len(euans_reviews))
    return euans_reviews, euans_labels

def pick_hyperparameters():
    pass
    

def train_model(load_path, save_path):
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=5, n_jobs=5, verbose=1)
    euans_data = pd.read_csv(load_path)
    X_train, y_train = split_data(euans_data)
    X_train = X_train[:32019]
    y_train = y_train[:32019]
    # there is probably something wrong with your pre-processing
    trained_model = grid_search.fit(X_train, y_train)
    print('training has finished !')
    save_path = save_path + '\gridsearch.joblib'
    print(save_path)
    joblib.dump(trained_model, save_path)
    
    
        

if __name__ == '__main__':
    saved_model_path = os.environ.get('SAVED_TRAINED_MODEL_PATH')
    loaded_data_path = os.environ.get('PROCESSED_TRAIN_DATA') 
    train_model(loaded_data_path, saved_model_path)