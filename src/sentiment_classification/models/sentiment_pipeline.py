from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import gc
import numpy as np
from transformers import Trainer, TrainerCallback
from torch import nn
import torch.nn.functional as F
import optuna
import pandas as pd
import yaml
import os
import sys
sys.path.append(os.getenv('LOCAL_ENV') + '/scripts')
print(sys.path)
from gpu_test import free_gpu_cache


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
                use_fast=True,
                max_length=512,
                problem_type="singe_label_classification")
            self.trainer = None
            self.label_mapping = {0:'negative', 1:'neutral', 2:'positive'}
            self.encoded_pred_labels = []
            self.decoded_pred_labels = []
            self.label_encoder = LabelEncoder()
            
    def optuna_hp_space(self, trial):
        '''
        Defines the hyperparameter space for Optuna.
        '''
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 3e-5, 2e-5, 3e-4, 1e-4]),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 64]),
            'per_device_eval_batch_size': trial.suggest_categorical('per_device_eval_batch_size', [4, 8, 64]),
            'num_train_epochs': trial.suggest_categorical('num_train_epochs', [10]),
        }
    
        
    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args['model_name_or_path'],
            num_labels=self.bert_params['num_of_labels'],
            max_length=512,
            problem_type="single_label_classification"
        )
        
    def extract_labels(self):
        for guess in self.encoded_pred_labels:
            for key, label in self.label_mapping.items():
                if guess == key:
                    self.decoded_pred_labels.append(label)
        return self.decoded_pred_labels
        
    def encode_labels(self, labels):
        self.encoded_pred_labels = [key for label in labels for key, value in self.label_mapping.items() if value == label]
        return np.array(self.encoded_pred_labels, dtype=np.int64)
    
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        logits = torch.Tensor(eval_pred.predictions)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.encoded_pred_labels = preds.tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=list(self.label_mapping.keys()))
        temp_precision, temp_recall, temp_f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        report_dict = {
            'precision': temp_precision,
            'recall': temp_recall,
            'f1 score': temp_f1,
        }
        final_report_dict = {
            'precision': {'Negative': precision[0], 'Neutral': precision[1],'Positive': precision[2]},
            'recall': {'Negative': recall[0], 'Neutral': recall[1],'Positive': recall[2]},
            'f1 score': {'Negative': f1[0], 'Neutral': f1[1],'Positive': f1[2]},
        }
        c_matrix = confusion_matrix(labels, preds, labels=list(self.label_mapping.keys()))

        report_df = pd.DataFrame(final_report_dict)
        c_matrix_df = pd.DataFrame(c_matrix, columns=list(self.label_mapping.values()), index=list(self.label_mapping.values()))
        c_matrix_df.to_csv(os.getenv('LOCAL_ENV') + '/logs/sentiment_classification/confusion_matrix.csv')
        report_df.to_csv(os.getenv('LOCAL_ENV') + '/logs/sentiment_classification/metrics_per_label.csv')

        return report_dict
                    
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


class MyTrainerCallback(TrainerCallback):
    memory_clear_interval = 1 

    @staticmethod
    def adjust_memory_clear_fraction():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        free_gpu_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        fraction = free_gpu_memory / total_gpu_memory
        return fraction

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.memory_clear_interval == 0:
            fraction = self.adjust_memory_clear_fraction()
            if fraction < 0.9:  
                torch.cuda.empty_cache()
                free_gpu_cache()
                gc.collect()