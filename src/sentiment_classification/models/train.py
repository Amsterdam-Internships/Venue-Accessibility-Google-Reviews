import os
import torch
from sentiment_pipeline import SentimentClassificationPipeline, MultiClassTrainer, EuansDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import TrainingArguments, DataCollatorWithPadding
from imblearn.under_sampling import RandomUnderSampler
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
import sys
sys.path.append(os.getenv('LOCAL_ENV') + '/scripts')
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from gpu_test import free_gpu_cache
import pandas as pd
import numpy as np
import yaml
import gc

# Load environment variables from .env file
load_dotenv(override=True)

# Load environment variables from .env file
config_path = os.getenv('LOCAL_ENV') + '/src/sentiment_classification/models/config.yml'
with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params['bert_params']

my_pipeline = SentimentClassificationPipeline(pipeline_type='transformer', model_type=params['model_name_or_path'])

def encode_datasets(train_text, val_text):
    new_train_encodings = my_pipeline.tokenizer(train_text,padding='max_length',truncation=True, max_length=512)
    new_val_encodings = my_pipeline.tokenizer(val_text, padding='max_length',truncation=True, max_length=512)
    return new_train_encodings, new_val_encodings

def create_datasets(euans_data):
    train_texts, val_texts, train_labels, val_labels = split_data(euans_data)
    train_encodings, val_encodings = encode_datasets(train_texts, val_texts)
    train_dataset = EuansDataset(train_encodings, train_labels)
    val_dataset = EuansDataset(val_encodings, val_labels)
    return train_dataset, val_dataset

def stratified_splitter(euans_reviews, encoded_labels):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_texts, val_texts, train_labels, val_labels = [], [], [], []

    for train_idx, val_idx in splitter.split(euans_reviews, encoded_labels):
        train_texts.extend([euans_reviews[i] for i in train_idx])
        val_texts.extend([euans_reviews[i] for i in val_idx])
        train_labels.extend([encoded_labels[i] for i in train_idx])
        val_labels.extend([encoded_labels[i] for i in val_idx])

    return train_texts, val_texts, train_labels, val_labels

def split_data(euans_data):
    euans_data = euans_data.rename(columns={"Sentiment": "labels"})
    euans_labels = euans_data.labels.values.tolist()
    encoded_labels = my_pipeline.encode_labels(euans_labels)
    euans_reviews = euans_data.Text.values.tolist()
    euans_reviews_array = np.array(euans_reviews)
    rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority', replacement=True)
    reviews_sampled, labels_sampled = rus.fit_resample(euans_reviews_array.reshape(-1, 1), encoded_labels)
    # Reshape the data back
    reviews_sampled = reviews_sampled.flatten().tolist()

    return stratified_splitter(reviews_sampled, labels_sampled)


def train_bert_models():
    # load the data
    euans_data = pd.read_csv(loaded_data_path)
    # split the data 
    train_dataset, val_dataset = create_datasets(euans_data)
    save_path = saved_model_path + f'/{names}'
    my_pipeline.training_args.output_dir = save_path
    data_collator = DataCollatorWithPadding(tokenizer=my_pipeline.tokenizer)
    
    # train the model
    my_pipeline.trainer = MultiClassTrainer(
        model=my_pipeline.model,
        args=my_pipeline.training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=my_pipeline.compute_metrics,
        tokenizer=my_pipeline.tokenizer,
        model_init=my_pipeline.model_init,
        data_collator=data_collator,
    )

    # Optimizing hyperparameters
    best_trial = my_pipeline.trainer.hyperparameter_search(
        direction='maximize',
        backend='optuna',
        hp_space=my_pipeline.optuna_hp_space,
        n_trials=10
    )
    best_parameters = best_trial.hyperparameters
        
    new_training_args = TrainingArguments(
        output_dir=save_path,
        logging_dir=logs_path,
        logging_strategy='epoch',
        logging_steps=10,
        # auto_find_batch_size=False,
        gradient_checkpointing=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        learning_rate=best_parameters['learning_rate'],
        per_device_train_batch_size=best_parameters['per_device_train_batch_size'],
        per_device_eval_batch_size=best_parameters['per_device_eval_batch_size'],
        # weight_decay=best_parameters['weight_decay'],
        num_train_epochs=best_parameters['num_train_epochs'],
        # gradient_accumulation_steps=best_parameters['gradient_accumulation_steps'],
        lr_scheduler_type=best_parameters['lr_scheduler_type'],
        load_best_model_at_end=True,
    )
    my_pipeline.trainer = MultiClassTrainer(
        model=my_pipeline.model,
        args=new_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=my_pipeline.compute_metrics,
    )
    # free_gpu_cache()
    # gc.collect()
    my_pipeline.trainer.train()
    # free_gpu_cache()
    # gc.collect()
    # device = my_pipeline.trainer.args.device  # Getting the device
    # torch.cuda.memory_summary(device=device, abbreviated=False)
    # print(f"Here Training device: {device}")
    print('Training of BERT models has finished!')
    save_path = saved_model_path + f"{names}"
    print(f'Saving model to {save_path}')
    my_pipeline.trainer.save_model(save_path)
    my_pipeline.tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    # Get the file paths from environment variables
    names = my_pipeline.model_name.split('/')[-1] if '/' in my_pipeline.model_name else my_pipeline.model_name
    loaded_data_path = os.getenv('LOCAL_ENV') + '/data/processed/aspect_classification_data/processed_euans_reviews.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + '/models/sentiment_classification/'
    logs_path = os.getenv('LOCAL_ENV') + '/logs/sentiment_classification/'
    train_bert_models()
