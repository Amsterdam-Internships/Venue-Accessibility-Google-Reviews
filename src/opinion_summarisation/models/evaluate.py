import os
import torch
import yaml
import pandas as pd
from rouge import Rouge
from sklearn.metrics import classification_report
from Pipelines import SummarizationPipeline

rouge = Rouge()

with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

pipeline = SummarizationPipeline(config['model_name'])

def load_test_data(test_data_path : str):
    google_data = pd.read_csv(test_data_path)
    encodings = pipeline.tokenizer(google_data['Text'].tolist(), truncation=True, padding=True)
    google_tensors = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']),
                                             torch.tensor(encodings['attention_mask']))
    return google_tensors

def load_ref_data(ref_data_path : str) -> list[str]:
    reference_df = pd.read_csv(ref_data_path)
    return reference_df.Text.values.tolist()
    
def compute_metrics(reference, predictions) -> dict:
    combined_ref = [' '.join(summary) for summary in reference]
    combined_pred = [' '.join(summary) for summary in predictions]
    scores = rouge.get_scores(combined_pred, combined_ref, avg=True)
    return scores
    

def evaluate_model(test_data_path : str, ref_data_path : str):
    google_dataset = load_test_data(test_data_path)
    data_loader = torch.utils.data.DataLoader(google_dataset, batch_size=config['batch_size'])
    pipeline.model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(pipeline.device), attention_mask.to(pipeline.device)
            outputs = pipeline.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                              max_length=config['max_summary_length'], num_beams=config['num_beams'],
                                              early_stopping=True)
            summaries = pipeline.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(summaries)
    reference_summaries = load_ref_data(ref_data_path)
    eval_metrics = pd.DataFrame(compute_metrics(reference_summaries, predictions))
    eval_metrics.to_json(results_path)


if __name__ == '__main__':
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_sentiment_labels.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation'
    results_path = os.getenv('LOCAL_ENV') + 'results/opinion_summarisation/'+config['model_name']+'.csv'
    
    pipeline.model = pipeline.model.from_pretrained(saved_model_path)

    evaluate_model(loaded_data_path)


    
