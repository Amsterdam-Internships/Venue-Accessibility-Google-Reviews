import os
import sys
import torch
import yaml
print(os.getenv('LOCAL_ENV'))
sys.path.append(os.getenv('LOCAL_ENV')+'/src')
import pandas as pd
import transformers
from dotenv import load_dotenv
from rouge import Rouge
from summarizer import Summarizer
from Pipelines import SummarizationPipeline
from opinion_summarisation.data.preprocessing import Preprocessor

preprocessor = Preprocessor()
rouge = Rouge()

# Load environment variables from .env file
load_dotenv()

config_path = os.getenv('LOCAL_ENV') + 'src/opinion_summarisation/models/config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

pipeline = SummarizationPipeline('distilbert-base-uncased', 'bert')

def generate_dataset(test_data_path: str):
    google_data = pd.read_csv(test_data_path)
    reviews = google_data['JoinedReview'].tolist()
    return reviews

def load_ref_data(ref_data_path: str) -> list[str]:
    reference_df = pd.read_csv(ref_data_path, delimiter=';')
    return reference_df['Summary'].values.tolist()

def compute_metrics(reference, predictions) -> dict:
    # Remove empty predictions
    non_empty_predictions = []
    non_empty_references = []
    for pred, ref in zip(predictions, reference):
        if pred.strip():  # Check if summary is not empty after removing leading/trailing whitespace
            non_empty_predictions.append(pred)
            non_empty_references.append(ref)
    scores = rouge.get_scores(non_empty_predictions, non_empty_references, avg=True)
    return scores


def evaluate_model(test_data_path: str, ref_data_path: str, results_path: str):
    google_dataset = generate_dataset(test_data_path)
    predictions = []
    tokenizer = pipeline.tokenizer
    model = pipeline.model
    if isinstance(pipeline.extractive_model, Summarizer):
        # Use the pipeline's extractive summarizer
        with torch.no_grad():
            print("Generating summaries using extractive summarizer...")
            for input_text in google_dataset:
                if preprocessor.count_sentences(input_text) == 1:
                    summary = input_text.replace('\n', '').strip()
                else:
                    summary = pipeline.extractive_model(input_text, num_sentences=1)
                
                predictions.append(summary)
                
        print("Finished generating summaries using extractive summarizer.")
        
    elif isinstance(model, transformers.BartForConditionalGeneration):
        # Load a saved BART model
        model = transformers.BartForConditionalGeneration.from_pretrained(saved_model_path)
        with torch.no_grad():
            print("Generating summaries using BART model...")
            for input_text in google_dataset:
                input_ids = tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=pipeline.max_length, return_tensors='pt')
                input_ids = input_ids.to(model.device)
                model.eval()  # Switch to evaluation mode
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=pipeline.max_summary_length,
                    num_beams=pipeline.num_beams,
                    early_stopping=True
                )
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(summary)
        print("Finished generating summaries using BART model.")
        
    else:
        raise ValueError("Unsupported model type.")

    reference_summaries = load_ref_data(ref_data_path)
    output_summaries = pd.DataFrame({'Prediction': predictions})
    eval_metrics = pd.DataFrame(compute_metrics(reference_summaries, predictions))
    output_summaries.to_csv(output_path, index=False)
    return eval_metrics

if __name__ == '__main__':
    loaded_data_path = os.getenv('LOCAL_ENV') + 'data/interim/grouped_reviews.csv'
    output_path = os.getenv('LOCAL_ENV') + 'data/external/output_summaries.csv'
    print('Evaluating model...')
    ref_data_path = os.getenv('LOCAL_ENV') + 'data/interim/ref_summaries.csv'
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation/'+ pipeline.model+'.bin'
    results_path = os.getenv('LOCAL_ENV') + 'results/opinion_summarisation/'+pipeline.model+'_eval_metrics.csv'
    metrics = evaluate_model(loaded_data_path, ref_data_path, results_path)
    metrics.to_csv(results_path, index=False)
    print('Done!')
    os._exit(0)
