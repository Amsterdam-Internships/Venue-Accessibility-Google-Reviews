from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
import time
import yaml
import os
import optuna
import csv

# Load environment variables from .env file
load_dotenv()
config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

class MyPipeline:
    def __init__(self, pipeline_type='default', bert_model=None):
        self.bert_params = params['bert_params']
        self.sk_params = params['sk_params']
        if pipeline_type == 'default':
            self.sk_pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('clf', VotingClassifier([
                    ('nb', MultinomialNB()),
                    ('lr', LogisticRegression())
                    ], voting=self.sk_params['clf']['voting'])
                )
            ])
            self.bert_pipeline = None
        elif bert_model is None:
            self.sk_pipeline = None
            self.bert_pipeline = None
        else:
            self.hyperparameters = []
            self.sk_pipeline = None
            self.model_name = bert_model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.num_labels = self.bert_params['num_of_labels']
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            self.best_params = None
            bert_classifier = Pipeline([
                ('tokenizer', self.tokenizer),
                (bert_model, self.model)
            ])
            self.best_loss = float('inf')
            self.bert_pipeline = bert_classifier
            self.classes = None
            self.label_binarizer = MultiLabelBinarizer()
            # Load BERT parameters from config file
            self.bert_epochs = self.bert_params['epochs']
            self.bert_batch_size = self.bert_params['batch_size']
            self.bert_learning_rate = self.format_params(self.bert_params['learning_rate'])
    
    def format_params(self, config):
        return [float(lr) for lr in config]
    
    def convert_to_tuple(self, subkey, subval):
        if subkey == 'ngram_range':
            return tuple([tuple(x) for x in subval])
        else:
            return subval
    
    def get_params(self, deep=True):
        return dict(sum([[(f'{key}__{subkey}', self.convert_to_tuple(subkey, subval)) for subkey, subval in val.items()] for key, val in self.sk_params.items()], []))

    def get_sk_pipeline(self):
        return self.sk_pipeline
    
    def get_bert_pipeline(self):
        return self.bert_pipeline
    
    def split_metrics(self, formatted_values):
        true_values =[]
        for y_true in formatted_values:
            true_values.append(y_true.split(','))
        return true_values
    
    def calculate_metrics(self, y_true, y_pred):
        true_values = self.split_metrics(y_true)
        accuracy = accuracy_score(true_values, y_pred, average='weighted')
        precision = precision_score(true_values, y_pred, average='weighted')
        recall = recall_score(true_values, y_pred, average='weighted')
        f1 = f1_score(true_values, y_pred, average='weighted')
        return accuracy, precision, recall, f1
            


    def fit(self, X, y, parameter_path):
        if self.sk_pipeline is not None:
            self.sk_pipeline.set_params(**self.params['default'])
            self.sk_pipeline.fit(X, y)
        if self.bert_pipeline is not None:
            self.model.to(self.device)

            # Prepare the dataset and dataloader
            encoded_texts = self.tokenizer(X, padding='max_length', truncation=True, return_tensors='pt', max_length=19)
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            y_binarised = self.label_binarizer.fit_transform(y)
            self.classes = self.label_binarizer.classes_
            labels_binarised = torch.tensor(y_binarised, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, labels_binarised)
            dataloader = DataLoader(dataset, batch_size=self.bert_batch_size[0], shuffle=True)

            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.objective(trial, X, y), n_trials=10)

            best_trial = study.best_trial
            best_params = best_trial.params
            best_learning_rate = best_params["learning_rate"]
            best_batch_size = best_params["batch_size"]
            best_num_epochs = best_params["num_epochs"]

            self.best_params = {
                "learning_rate": best_learning_rate,
                "batch_size": best_batch_size,
                "num_epochs": best_num_epochs,
            }

            print('Started ...')
            start_time = time.time()

            self.best_loss = self.train_model(dataloader, best_learning_rate, best_num_epochs, best_batch_size)

            end_time = time.time()
            total_time = end_time - start_time
            epoch_time = total_time / best_num_epochs
            print("Time for all epochs:", epoch_time * best_num_epochs)
            self.save_hyperparameters_to_csv(self.hyperparameters, parameter_path + 'hyperparameters.csv')

        return self


    def predict(self, X, model_path):
        if self.sk_pipeline is not None:
            return self.sk_pipeline.predict(X)
        elif self.bert_pipeline is not None:
            try:
                encoded_texts = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded_texts['input_ids'].to(self.device)
                attention_mask = encoded_texts['attention_mask'].to(self.device)
                
                # Load the fine-tuned model from a file
                fine_tuned_model = torch.load(model_path)
                fine_tuned_model.to(self.device)
                
                outputs = fine_tuned_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Apply sigmoid activation function to obtain probabilities
                probabilities = F.sigmoid(logits)

                # Set a threshold to determine the predicted labels
                threshold = 0.75
                predicted_labels = (probabilities > threshold).int().tolist()

                # Convert the predicted labels to a list of aspects for each input sample
                predictions = []
                for labels in predicted_labels:
                    prediction = [self.classes[i] for i, label in enumerate(labels) if label==1]
                    predictions.append(prediction)
                return predictions
            except Exception as e:
                # Handle specific exceptions or log the error
                raise RuntimeError("Error occurred during BERT prediction: " + str(e))
        else:
            raise ValueError('Both pipelines are None. Please provide a valid pipeline type.')
