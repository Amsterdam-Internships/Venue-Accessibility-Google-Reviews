from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
import torch
from transformers import Trainer, TrainerCallback
from torch import nn
import yaml
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.getenv('LOCAL_ENV') + '/scripts')
print(sys.path)
from gpu_test import free_gpu_cache


config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'

class AspectClassificationPipeline:
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
                problem_type="multi_label_classification")
            self.training_args = TrainingArguments(
                output_dir='./results',
                learning_rate=self.bert_params['learning_rate'],
                per_device_eval_batch_size=self.bert_params['batch_size'],
                num_train_epochs=self.bert_params['epochs'],
                evaluation_strategy='epoch') 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args['model_name_or_path'],
                use_fast=True,
                max_length=512,
                problem_type="multi_label_classification",
                truncation=True)
            self.trainer = None
            self.label_binarizer = MultiLabelBinarizer()
            self.label_mapping = {0: 'Access', 1: 'Overview', 2: 'Staff', 3: 'Toilets', 4: 'Transport & Parking'}
            self.encoded_pred_lables = []
            self.decoded_pred_labels = []
            
    def optuna_hp_space(self, trial):
        '''
        Defines the hyperparameter space for Optuna.
        '''
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 4e-5, 3e-5, 2e-5]),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16]),
            'per_device_eval_batch_size': trial.suggest_categorical('per_device_eval_batch_size', [4, 8, 16]),
            'num_train_epochs': trial.suggest_categorical('num_train_epochs', [2, 3, 4, 5]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 3, 4])
        }
        
    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args['model_name_or_path'],
            num_labels=self.bert_params['num_of_labels'],
            max_length=512,
            problem_type="multi_label_classification"
        )
        
    def extract_labels(self):
        for guess in self.encoded_pred_lables:
            sublist = []
            for i, value in enumerate(guess):
               if value == 1:
                sublist.append(self.label_mapping[i])
            self.decoded_pred_labels.append(sublist)
        return self.decoded_pred_labels

    def find_best_threshold(self, labels, preds, thresholds, metric='f1'):
        best_metric_value = 0
        best_threshold = 0

        for threshold in thresholds:
            pred_labels = (preds > threshold).float()
            precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='weighted', labels=list(self.label_mapping.keys()))

            # Choose the metric to optimize
            if metric == 'precision':
                metric_value = precision
            elif metric == 'recall':
                metric_value = recall
            elif metric == 'f1':
                metric_value = f1
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if  metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        return best_threshold
    
       
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        logits = eval_pred.predictions
        preds = torch.sigmoid(torch.Tensor(logits))
        # Threshold tuning
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # You can adjust the range
        best_threshold = self.find_best_threshold(labels, preds, thresholds, metric='f1')

        # Apply the best threshold to get final predicted labels
        pred_labels = (preds > best_threshold).float()
        self.encoded_pred_lables = pred_labels
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average=None, labels=list(self.label_mapping.keys()))
        temp_precision, temp_recall, temp_f1, _ = precision_recall_fscore_support(labels, pred_labels, average='weighted', labels=list(self.label_mapping.keys()))
        report_dict = {
            'precision': temp_precision,
            'recall': temp_recall,
            'f1 score': temp_f1,
        }
        final_report_dict = {
            'precision': {'Access': precision[0], 'Overview': precision[1],'Staff': precision[2],'Toilets': precision[3],'Transport & Parking': precision[4]},
            'recall': {'Access': recall[0], 'Overview': recall[1],'Staff': recall[2],'Toilets': recall[3],'Transport & Parking': recall[4]},
            'f1 score': {'Access': f1[0], 'Overview': f1[1],'Staff': f1[2],'Toilets': f1[3],'Transport & Parking': f1[4]},
        }
        
        # metrics_df = pd.DataFrame(classification_report(labels, pred_labels, output_dict=True))   

        # metrics_df.to_csv(os.getenv('LOCAL_ENV') + '/logs/aspect_classification/metrics_per_label.csv')

        report_df = pd.DataFrame(final_report_dict, index=list(self.label_mapping.values()))
        report_df.to_csv(os.getenv('LOCAL_ENV') + '/logs/aspect_classification/metrics_per_label.csv')

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
    
                  
class MultiLabelClassTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_predictions = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    
       
class MyTrainerCallback(TrainerCallback):
    memory_clear_interval = 5
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.memory_clear_interval == 0:
            torch.cuda.empty_cache()
            free_gpu_cache()