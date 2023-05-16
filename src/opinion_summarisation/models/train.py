from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from Pipelines import SummarizationPipeline
import pandas as pd
import torch
import yaml
import os



with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


pipeline = SummarizationPipeline(config['model_name'])

def train_bert_models():
    
    dataset = pd.read_csv(loaded_data_path)  
    
    train_data, val_data = train_test_split(dataset, test_size=config['validation_split'], random_state=42)

    # Tokenize the input reviews
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    train_encodings = tokenizer(train_data['Review'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_data['Review'].tolist(), truncation=True, padding=True)

    # Prepare the data as tensors
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                  torch.tensor(train_encodings['attention_mask']))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_encodings['input_ids']),
                                                torch.tensor(val_encodings['attention_mask']))

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])

    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(pipeline.model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(config['num_epochs']):
        pipeline.model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(pipeline.device), attention_mask.to(pipeline.device)
            outputs = pipeline.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # Evaluate on the validation set
        pipeline.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(pipeline.device), attention_mask.to(pipeline.device)
                outputs = pipeline.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                val_loss += loss.item()
            val_loss /= len(val_loader)

        # Print the training progress
        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the fine-tuned model
    pipeline.model.save_pretrained(saved_model_path)

if __name__ == '__main__':
    # Get the file paths from environment variables
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_sentiment_labels.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation'
