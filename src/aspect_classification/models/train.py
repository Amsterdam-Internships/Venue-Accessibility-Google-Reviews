from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from transformers import TrainingArguments
from aspect_pipeline import AspectClassificationPipeline, EuansDataset, MultiLabelClassTrainer, MyTrainerCallback
# Load environment variables from .env file
load_dotenv(override=True)
import gc
import pandas as pd
import numpy as np
import joblib
import yaml
import torch
import sys
import os
sys.path.append(os.getenv('LOCAL_ENV') + '/scripts')
print(sys.path)
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from aspect_classification.data.preprocessing import Preprocessor
config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'
my_trainer_callback = MyTrainerCallback()

with open(config_path, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
my_pipeline = AspectClassificationPipeline(pipeline_type='transformer', model_type=params['bert_params']['model_name_or_path'])
processor = Preprocessor()
custom_trainer = MultiLabelClassTrainer(model=my_pipeline.model)
torch.cuda.set_per_process_memory_fraction(0.5)  # Adjust as needed
torch.backends.cudnn.benchmark = True

def encode_datasets(train_text, val_text):
    new_train_encodings = my_pipeline.tokenizer(train_text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    new_val_encodings = my_pipeline.tokenizer(val_text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    return new_train_encodings, new_val_encodings

def create_datasets(euans_data):
    train_texts, val_texts, train_labels, val_labels = split_data(euans_data)
    train_encodings, val_encodings = encode_datasets(train_texts, val_texts)
    train_dataset = EuansDataset(train_encodings, train_labels)
    val_dataset = EuansDataset(val_encodings, val_labels)
    return train_dataset, val_dataset

def split_data(euans_data):
    euans_data = euans_data.rename(columns={"Aspect": "labels"})
    euans_labels = [[label] for label in euans_data.labels.values.tolist()]
    euans_labels = my_pipeline.label_binarizer.fit_transform(euans_labels)
    euans_labels = euans_labels.astype(np.float32)
    euans_reviews = euans_data.Text.values.tolist()
    return train_test_split(euans_reviews, euans_labels, test_size=.2)

def train_bert_models():
    name = my_pipeline.model_name.split('/')[-1] if '/' in my_pipeline.model_name else my_pipeline.model_name
    save_path = saved_model_path+f"{name}"
    # load the data
    try:
        euans_data = pd.read_csv(loaded_data_path)
    except AttributeError as e:
        print(f"{e.name=}") 
    # split the data 
    train_dataset, val_dataset = create_datasets(euans_data)
    my_pipeline.training_args.output_dir = save_path
    print(f"my device {my_pipeline.device}")
    print(f"my model {my_pipeline.model_name}")
    
    #train the model
    my_pipeline.trainer = MultiLabelClassTrainer(
        model=my_pipeline.model,
        args = my_pipeline.training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics=my_pipeline.compute_metrics,
        tokenizer=my_pipeline.tokenizer,
        model_init=my_pipeline.model_init,
    )
    # optimising hyperparameters
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
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=best_parameters['learning_rate'],
        auto_find_batch_size=False,
        gradient_checkpointing=True,
        fp16=True,
        load_best_model_at_end=True,
        per_device_train_batch_size=best_parameters['per_device_train_batch_size'],
        per_device_eval_batch_size=best_parameters['per_device_eval_batch_size'],
        num_train_epochs=best_parameters['num_train_epochs'],
        gradient_accumulation_steps=best_parameters['gradient_accumulation_steps']
    )
    print(torch.cuda.get_device_properties(0))
    my_pipeline.trainer = MultiLabelClassTrainer(
        model = my_pipeline.model,
        args=new_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=my_pipeline.compute_metrics,
        callbacks=[my_trainer_callback],
    )
    try:
        my_pipeline.trainer.train()
    except Exception as e:
        print("CUDA runtime error:", e)
    device = my_pipeline.trainer.args.device  # Getting the device
    torch.cuda.memory_summary(device=device, abbreviated=False)
    print(f"Here Training device: {device}")
    print('Training of BERT models has finished!')
    my_pipeline.trainer.save_model(save_path)
    my_pipeline.tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    # Get the file paths from environment variables
    loaded_data_path = os.getenv('LOCAL_ENV') + '/data/processed/aspect_classification_data/processed_euans_reviews.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + '/models/aspect_classification/transformer_models/'
    params_path = os.getenv('LOCAL_ENV') + 'models/aspect_classification/transformer_models/'
    logs_path = os.getenv('LOCAL_ENV') + '/logs/aspect_classification/'
    train_bert_models()
