import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from summarizer import Summarizer,TransformerSummarizer
import transformers
from sklearn.pipeline import Pipeline

class SummarizationPipeline:
    def __init__(self, max_length=1024, num_beams=4, max_summary_length=100, model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_beams = num_beams
        self.max_summary_length = max_summary_length
        
    def __call__(self, text):
        inputs = self.tokenizer(text, max_length=self.max_length, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=self.num_beams, max_length=self.max_summary_length, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        

      

