from sklearn.model_selection import train_test_split
from Pipelines import SummarizationPipeline
import pandas as pd
import torch
import yaml
import os

with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

pipeline = SummarizationPipeline(config['model_name'])

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

    optimizer = torch.optim.AdamW(pipeline.model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        pipeline.model.train()
        train_loss = 0
        for batch in euans_loader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(pipeline.device), attention_mask.to(pipeline.device)
            outputs = pipeline.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(euans_loader)

        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}")

    pipeline.model.save_pretrained(saved_model_path)

if __name__ == '__main__':
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_sentiment_labels.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation/' + config['model_name'] + '.bin'
