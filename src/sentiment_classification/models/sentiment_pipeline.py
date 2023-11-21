from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import numpy as np
from transformers import Trainer
from torch import nn
import optuna
import yaml
import os


config_path = os.getenv('LOCAL_ENV') + '/src/sentiment_classification/models/config.yml'

class SentimentClassificationPipeline:
    def __init__(self, pipeline_type='default', model_type=None):
        with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.bert_params = params['bert_params']
        if pipeline_type == 'transformer':
            self.model_name = self.bert_params['model_name_or_path']
            self.model_args = {'model_name_or_path': self.model_name}
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_args['model_name_or_path'],
                num_labels=self.bert_params['num_of_labels'],
                problem_type="single_label_classification")
            self.training_args = TrainingArguments(
                output_dir='./results/sentiment_classification',
                learning_rate=self.bert_params['learning_rate'],
                per_device_eval_batch_size=self.bert_params['batch_size'],
                num_train_epochs=self.bert_params['epochs'],
                evaluation_strategy='epoch') 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args['model_name_or_path'],
                max_length=512,
                problem_type="singe_label_classification")
            self.trainer = None
            self.label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
            self.encoded_sent_labels = []
            self.decoded_sent_labels = []
            
    def optuna_hp_space(self, trial):
        '''
        Defines the hyperparameter space for Optuna.
        '''
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16, 32]),
            'per_device_eval_batch_size': trial.suggest_categorical('per_device_eval_batch_size', [4, 8, 16, 32]),
            'num_train_epochs': trial.suggest_categorical('num_train_epochs', [2, 3, 4, 5]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 3, 4]),
        }
        
    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args['model_name_or_path'],
            num_labels=self.bert_params['num_of_labels'],
            max_length=512,
            problem_type="single_label_classification"
        )
        
    def encode_labels(self, labels):
        self.encoded_sent_labels = [self.label_mapping[label] for label in labels]
        return np.array(self.encoded_sent_labels, dtype=np.int64)
    
    def decode_labels(self, class_probabilities):
        encoded_labels = class_probabilities.argmax(dim=1)
        self.decoded_sent_labels = [list(self.label_mapping.keys())[list(self.label_mapping.values()).index(encoded_label in encoded_labels)]]
        return self.decoded_sent_labels
        
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        logits = torch.Tensor(eval_pred.predictions)
        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        return {"f1 score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy}

                
class EuansDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        encoding = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        encoding['labels'] = torch.tensor(self.labels[idx])
        return encoding

    def __len__(self):
        return len(self.labels)
                

  
class MultiClassTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):        
        labels = inputs.get("labels")
        labels = labels.to(torch.long)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss
