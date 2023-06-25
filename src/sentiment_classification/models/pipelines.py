from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from dotenv import load_dotenv
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import yaml
# Load environment variables from .env file
load_dotenv()
config_path = os.getenv('LOCAL_ENV') + '/src/sentiment_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)



class SentimentPipeline:
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
            # self.bert_pipeline = None
        elif bert_model is None:
            self.sk_pipeline = None
            self.bert_pipeline = None
        else:
            self.model_name = bert_model
            self.label_mapping = {0: 'Negative', 1: 'Positive'}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.num_labels = self.bert_params['num_of_labels']
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=self.num_labels)
            bert_classifier = Pipeline([
                ('tokenizer', self.tokenizer),
                (bert_model, self.model)
            ])
            self.sk_pipeline = None
            self.bert_pipeline = bert_classifier
            self.label_encoder = LabelEncoder()
            self.label_binarizer = LabelBinarizer()

            # Load BERT parameters from config file
            self.bert_epochs = self.bert_params['epochs']
            self.bert_batch_size = self.bert_params['batch_size']
            self.bert_learning_rate = self.format_params(self.bert_params['learning_rate'])
    def format_params(self, params):
        return [float(lr) for lr in params]
    
    def update_label_mapping(self, label_encoder):
        self.label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
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
    
    def calculate_metrics(self, y_true, y_pred):
        # Assuming 'column_name' is the name of the column you want to check
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        if np.isclose(precision, 0.0):
            labels_with_zero_precision = np.unique(y_pred)
            for label_idx in labels_with_zero_precision:
                print("Label with zero precision:", label_idx)
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        return accuracy, precision, recall, f1
            
        
    
    def evaluate(self, y_true, y_pred):
        # Calculate metrics

        # for tru_val, pred_val in zip(y_true, y_pred):
        #    print(type(tru_val), type(pred_val))

        accuracy, precision, recall, f1 = self.calculate_metrics(y_true, y_pred)
        evaluation_metrics = {'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1}
        
        return evaluation_metrics

    
    
    
    def fit(self, X, y):
        if self.sk_pipeline is not None:
            self.sk_pipeline.set_params(**self.params['default'])
            self.sk_pipeline.fit(X, y)
        if self.bert_pipeline is not None:
            encoded_texts = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            y_encoded = self.label_encoder.fit_transform(y)
            self.update_label_mapping(self.label_encoder)
            labels_encoded = torch.tensor(y_encoded, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, labels_encoded)
            dataloader = DataLoader(dataset, batch_size=self.bert_batch_size[0], shuffle=True)
            optimizer = optim.AdamW(self.model.parameters(), lr=self.bert_learning_rate[2])
            start_time = time.time()
            print('Started ...')
            for epoch in range(self.bert_epochs[0]):
                for batch in dataloader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.bert_learning_rate[2])
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    criterion = nn.CrossEntropyLoss()
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                end_time = time.time()
                epoch_time = end_time - start_time
            print("Time for all epochs:", epoch_time*self.bert_epochs[0])
        self.update_label_mapping(self.label_encoder)
        return self

    def predict(self, X):
        if self.sk_pipeline is not None:
            return self.sk_pipeline.predict(X)
        elif self.bert_pipeline is not None:
            try:
                encoded_texts = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded_texts['input_ids'].to(self.device)
                attention_mask = encoded_texts['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, y_pred = torch.max(outputs[0], dim=1)
                y_pred = y_pred.cpu().numpy()
                # Convert numerical predictions to text labels
                y_pred_labels = [self.label_mapping[label] for label in y_pred]
                return y_pred_labels
            except Exception as e:
                # Handle specific exceptions or log the error
                raise RuntimeError("Error occurred during BERT prediction: " + str(e))
        else:
            raise ValueError('Both pipelines are None. Please provide a valid pipeline type.')
