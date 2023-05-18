import os
import torch
import yaml
import pandas as pd
from rouge import Rouge
import transformers
from summarizer import Summarizer
from Pipelines import SummarizationPipeline

rouge = Rouge()

with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# clean this up later
se2seq_config = config['se2seq_config']
bert_config = config['berts']['bert_config']
distilbert_config = config['berts']['distilbert_config']
sbert_config = config['berts']['sbert_config']

pipeline = SummarizationPipeline('bert', bert_config['model_name'])

def load_test_data(test_data_path : str) -> torch.utils.data.TensorDataset:
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

def evaluate_model(test_data_path: str, ref_data_path: str):
    google_dataset = load_test_data(test_data_path)
    data_loader = torch.utils.data.DataLoader(google_dataset, batch_size=config['se2seq_config']['batch_size'])
    pipeline.model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(pipeline.extractive_model, Summarizer):
                # BERT Extractive Summarizer model
                input_ids = batch[0]
                input_ids = input_ids.to(pipeline.device)
                summaries = pipeline.extractive_model(input_ids)
            elif isinstance(pipeline.model, transformers.BartForConditionalGeneration):
                # Seq2Seq model
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(pipeline.device), attention_mask.to(pipeline.device)
                outputs = pipeline.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=pipeline.max_summary_length,
                    num_beams=pipeline.num_beams,
                    early_stopping=True
                )
                summaries = pipeline.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                raise ValueError("Unsupported model type.")

            predictions.extend(summaries)

    reference_summaries = load_ref_data(ref_data_path)
    eval_metrics = pd.DataFrame(compute_metrics(reference_summaries, predictions))
    eval_metrics.to_csv(results_path, index=False)




if __name__ == '__main__':
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_sentiment_labels.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation/bertmodel.bin'
    results_path = os.getenv('LOCAL_ENV') + 'results/opinion_summarisation/'+bert_config['model_name']+'.csv'
    
    pipeline.model = pipeline.model.from_pretrained(saved_model_path)

    evaluate_model(loaded_data_path, results_path)
