from Pipelines import SummarizationPipeline
import sys 
import os 
from dotenv import load_dotenv
sys.path.append(os.getenv('LOCAL_ENV')+'/src')
import pandas as pd
from summarizer import Summarizer
import transformers
import torch
import yaml
import os

# Load environment variables from .env file
load_dotenv()

config_path = os.getenv('LOCAL_ENV') + '/src/opinion_summarisation/models/config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pipeline = SummarizationPipeline('distilbert-base-uncased', 'distilbert')
config = pipeline.config

def load_data(loaded_data_path):
    euans_data = pd.read_csv(loaded_data_path)  
    euans_encodings = pipeline.tokenizer(euans_data['Text'].tolist(), truncation=True, padding=True)
    
    # Prepare the data as tensors
    euans_tensors = torch.utils.data.TensorDataset(torch.tensor(euans_encodings['input_ids']),
                                                  torch.tensor(euans_encodings['attention_mask']))
    
    return euans_tensors

def train_bert_models():
    euans_dataset = load_data(loaded_data_path)
    euans_loader = torch.utils.data.DataLoader(euans_dataset, batch_size=config['batch_size'], shuffle=True)
    summaries = []
    
    if isinstance(pipeline.extractive_model, Summarizer):
        print("Model is an extractive summarizer. No training required.")
        return
    
    for epoch in range(config['num_epochs']):
        pipeline.model.train()
        train_loss = 0
        for batch in euans_loader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(pipeline.model.device), attention_mask.to(pipeline.model.device)
            
            if isinstance(pipeline.model, transformers.BartForConditionalGeneration):
                optimizer = torch.optim.AdamW(pipeline.model.parameters(), lr=config['learning_rate'])
                
                outputs = pipeline.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                summaries_batch = pipeline.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Perform any necessary operations with the summaries
                summaries.extend(summaries_batch)
                
                optimizer.zero_grad()
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            else:
                raise ValueError("Unsupported model type.")
        
        train_loss /= len(euans_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}")
    
    if isinstance(pipeline.model, transformers.BartForConditionalGeneration):
        pipeline.model.save_pretrained(saved_model_path, save_config=True)
        

            
if __name__ == '__main__':
    saved_model_path = None
    if pipeline.config['model_name'] == "facebook/bart-large":
        names = pipeline.config['model_name'].split("/")
        
        saved_model_path = os.getenv('LOCAL_ENV') + f'/models/opinion_summarisation/{names[1]}' + '.bin'
    else:
        name = pipeline.config['model_name']
        saved_model_path = os.getenv('LOCAL_ENV') + f'/models/opinion_summarisation/{name}' + '.bin'
        print('saved')
    loaded_data_path = os.getenv('LOCAL_ENV') + '/data/processed/aspect_classification_data/processed_euans_reviews.csv'
    train_bert_models()

