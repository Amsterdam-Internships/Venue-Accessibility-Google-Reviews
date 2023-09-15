from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from transformers.trainer_utils import PredictionOutput
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from transformers import Trainer
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_metric
from optuna import trial
import numpy as np
import yaml
import os


config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'

class AspectClassificationPipeline:
    def __init__(self, pipeline_type='default', model_type=None):
        with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.bert_params = params['bert_params']
        self.sk_params = params['sk_params']
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
                max_length=512,
                problem_type="multi_label_classification")
            self.trainer = None
            self.label_binarizer = MultiLabelBinarizer()
            self.label_mapping = {0: 'Access', 1: 'Overview', 2: 'Staff', 3: 'Toilets', 4: 'Transport & Parking'}
            
    def optuna_hp_space(self, trial):
        '''
        Defines the hyperparameter space for Optuna.
        '''
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32, 64]),
            'num_train_epochs': trial.suggest_categorical('num_train_epochs', [2, 3, 4, 5]),
        }
        
    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args['model_name_or_path'],
            num_labels=self.bert_params['num_of_labels'],
            max_length=512,
            problem_type="multi_label_classification"
        )
        
    def extract_labels(self, predicted_probabilities):
        label_text = self.label_binarizer.classes_
        label_index = np.argmax(predicted_probabilities, axis=1)
        predicted_labels_text = [label_text[i] for i in label_index]
        return predicted_labels_text

        
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        # logits = torch.Tensor(eval_pred.predictions)
        # outputs = torch.relu(logits)
        # preds = F.sigmoid(outputs) # This is correct because each label event is independent
        # threshold = 0.5 #correct threshold to use when using the sigmoid activation function
        logits = eval_pred.predictions
        preds = torch.sigmoid(torch.Tensor(logits))
        threshold = 0.5
        probs = (preds > threshold).float()
        f1 = f1_score(labels, probs, average='samples')
        precision = precision_score(labels, probs, average='samples')
        recall = recall_score(labels, probs, average='samples')
        return {"f1 score": f1,
                "precision": precision,
                "recall": recall}
                        
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
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        # logits = outputs.get("logits")
        logits = outputs.logits
        # probabilities = torch.sigmoid(logits)
        loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    
    # def prediction_step(self, model: Module, inputs: Dict[str, Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[Tensor | None, Tensor | None, Tensor | None]:
    #     return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

