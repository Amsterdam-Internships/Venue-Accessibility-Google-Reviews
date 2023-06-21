from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.model_name = bert_model
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
    
    def format_metrics(self, y_true, y_pred):
        formatted_values = []
        for true_val, pred_row in zip(y_true, y_pred):
            for pred_val in pred_row:
                formatted_values.append((true_val, pred_val))
        return formatted_values
    
    def split_metrics(self, formatted_values):
        true_values =[]
        predicted_values = []
        for y_true, y_pred in formatted_values:
            true_values.append(y_true)
            predicted_values.append(y_pred)
        return true_values, predicted_values
    
    
    def calculate_metrics(self, formatted_values):
        true_values, predicted_values = self.split_metrics(formatted_values)
        accuracy = accuracy_score(true_values, predicted_values)
        precision = precision_score(true_values, predicted_values, average='macro')
        recall = recall_score(true_values, predicted_values, average='macro')
        f1 = f1_score(true_values, predicted_values, average='macro')
        return accuracy, precision, recall, f1
            
        
    
    def evaluate(self, y_true, y_pred):
        # Calculate metrics
        formatted_values = self.format_metrics(y_true, y_pred)
        accuracy, precision, recall, f1 = self.calculate_metrics(formatted_values)
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
            encoded_texts = self.tokenizer(X, padding='max_length', truncation=True, return_tensors='pt', max_length=19)
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            y_encoded = self.label_encoder.fit_transform(y)  
            # Adjust label encoding
            labels_encoded = torch.tensor(y_encoded, dtype=torch.long)  # Convert to torch.long type

            dataset = TensorDataset(input_ids, attention_mask, labels_encoded)
            dataloader = DataLoader(dataset, batch_size=self.bert_batch_size[0], shuffle=True)  # Add shuffle for better training

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.bert_learning_rate[2])
            criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class labels
            start_time = time.time()

            print('Started ...')
            for epoch in range(self.bert_epochs[0]):
                for batch in dataloader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                end_time = time.time()
                epoch_time = end_time - start_time
            print("Time for all epochs:", epoch_time * self.bert_epochs[0])
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
                logits = outputs.logits  # Get the predicted logits

                # Apply sigmoid activation function to obtain probabilities
                probabilities = F.sigmoid(logits)

                # Set a threshold to determine the predicted labels
                threshold = 0.5
                predicted_labels = (probabilities > threshold).int().tolist()

                # Define the aspect labels
                aspect_labels = ['Access', 'Overview', 'Staff', 'Toilets', 'Transport & Parking']

                # Convert the predicted labels to a list of aspects for each input sample
                predictions = []
                for labels in predicted_labels:
                    prediction = [aspect_labels[i] for i, label in enumerate(labels) if label==1]
                    predictions.append(prediction)
                    
                return predictions
            except Exception as e:
                # Handle specific exceptions or log the error
                raise RuntimeError("Error occurred during BERT prediction: " + str(e))
        else:
            raise ValueError('Both pipelines are None. Please provide a valid pipeline type.')






