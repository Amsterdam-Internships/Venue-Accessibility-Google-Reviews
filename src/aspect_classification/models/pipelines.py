from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from sklearn.metrics import hamming_loss, jaccard_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from dotenv import load_dotenv
import numpy as np
import torch
import json
import time
import yaml
import os 

# Load environment variables from .env file
load_dotenv()
config_path = os.getenv('LOCAL_ENV') + 'src/aspect_classification/models/config.yml'
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
            # self.bert_pipeline = None
        elif bert_model is None:
            self.sk_pipeline = None
            self.bert_pipeline = None
        else:
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
            self.label_binarizer = MultiLabelBinarizer()
            # Load BERT parameters from config file
            self.bert_epochs = self.bert_params['epochs']
            self.bert_batch_size = self.bert_params['batch_size']
            self.bert_learning_rate = self.format_params(self.bert_params['learning_rate'])
    def format_params(self, params):
        return [float(lr) for lr in params]
    
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
    
    def evaluate(self, y_true, y_pred):
        evaluation_metrics = {}
        # Compute and store evaluation metrics.
        # This is just numpy arrays so just use the prediction scores.
        evaluation_metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        evaluation_metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples')
        evaluation_metrics['f1_score'] = f1_score(y_true, y_pred, average='samples')
        return evaluation_metrics
    
    def fit(self, X, y):
        if self.sk_pipeline is not None:
            self.sk_pipeline.set_params(**self.params['default'])
            self.sk_pipeline.fit(X, y)
        if self.bert_pipeline is not None:
            encoded_texts = self.tokenizer(X, padding='max_length', truncation=True, return_tensors='pt', max_length=19)
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            y_encoded = self.label_binarizer.fit_transform(y)  # Use MultiLabelBinarizer

            # Adjust label encoding
            labels_encoded = torch.tensor(y_encoded, dtype=torch.float)  # Convert to torch.float type

            dataset = TensorDataset(input_ids, attention_mask, labels_encoded)
            dataloader = DataLoader(dataset, batch_size=self.bert_batch_size[0], shuffle=True)  # Add shuffle for better training

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.bert_learning_rate[2])
            start_time = time.time()

            print('Started ...')
            for epoch in range(self.bert_epochs[0]):
                for batch in dataloader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    loss.backward()
                    optimizer.step()
                end_time = time.time()
                epoch_time = end_time - start_time
            print("Time for all epochs:", epoch_time*self.bert_epochs[0])
        return self

    def predict(self, X):
        if self.sk_pipeline is not None:
            return self.sk_pipeline.predict(X)
        elif self.bert_pipeline is not None:
            encoded_texts = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded_texts['input_ids'].to(self.device)
            attention_mask = encoded_texts['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Get the predicted logits

            # Apply sigmoid activation function to obtain probabilities
            probabilities = torch.sigmoid(logits)
            # Set a threshold to determine the predicted labels
            threshold = 0.5
            predicted_labels = (probabilities > threshold).int()
            # Define the aspect labels
            aspect_labels = ["Toilets", "Transport & Parking", "Accessibility", "Staff", "Overview"]

            # Convert the predicted labels to a list of aspects for each input sample
            aspects = []
            for labels in predicted_labels:
                indices = np.where(labels == 1)[0]
                aspects.append([aspect_labels[i] for i in indices])

            return aspects
        else:
            raise ValueError('Both pipelines are None. Please provide a valid pipeline type.')





