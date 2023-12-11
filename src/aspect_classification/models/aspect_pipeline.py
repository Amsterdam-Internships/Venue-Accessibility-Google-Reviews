from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from transformers import Trainer, TrainerCallback
from torch import nn
import yaml
import os
import sys
sys.path.append(os.getenv('LOCAL_ENV') + '/scripts')
print(sys.path)
from gpu_test import free_gpu_cache
memory_clear_interval = 5

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

        
    def compute_metrics(self, eval_pred):
        labels = eval_pred.label_ids
        # correct threshold to use when using the sigmoid activation function
        logits = eval_pred.predictions
        preds = torch.sigmoid(torch.Tensor(logits))
        threshold = 0.5
        pred_labels = (preds > threshold).float()
        self.encoded_pred_lables = pred_labels
        f1 = f1_score(labels, pred_labels, average='weighted')
        precision = precision_score(labels, pred_labels, average='weighted')
        recall = recall_score(labels, pred_labels, average='weighted')
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
        self.all_predictions = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    
       
class MyTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % memory_clear_interval == 0:
            torch.cuda.empty_cache()
            free_gpu_cache()