from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import torch
import numpy as np
from transformers import Trainer
from torch import nn
import torch.nn.functional as F
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
            'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 4e-5, 3e-5, 2e-5]),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32, 64, 128]),
            'per_device_eval_batch_size': trial.suggest_categorical('per_device_eval_batch_size', [8, 16, 32, 64, 128]),
            'num_train_epochs': trial.suggest_categorical('num_train_epochs', [4, 5, 6, 7, 8, 9, 10]),
            # 'L2_reg': trial.suggest_float("L2_reg", 1e-3, 1e-2, log=True)
            # 'hidden_dropout_prob': trial.suggest_categorical('hidden_dropout_prob', [0.1, 0.2, 0.3, 0.4, 0.5])
            # 'weight_decay': trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True)
            # 'lr_scheduler_type': trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine', 'constant'])
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
        self.encoded_pred_labels = [self.label_mapping[label] for label in labels]
        return np.array(self.encoded_pred_labels, dtype=np.int64)
    
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        logits = torch.Tensor(eval_pred.predictions)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.encoded_pred_labels = preds.tolist()
        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)

        return {"f1 score": f1,
                "accuracy": accuracy,
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
                

  
class MultiClassTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):        
        labels = inputs.get("labels")
        labels = labels.to(torch.long)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # class_labels = np.unique(labels.cpu().numpy())
        # class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y = labels.cpu().numpy())
        # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        # loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss
