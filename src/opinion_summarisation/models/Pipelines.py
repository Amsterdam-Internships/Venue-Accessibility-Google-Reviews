import yaml
import torch
import transformers
from summarizer import Summarizer

with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class SummarizationPipeline:
    def __init__(self, model_type, model_name):
        if model_type == 'seq2seq':
            self.seq2seq_config = config['seq2seq_config']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.seq2seq_config['model_name'])
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.seq2seq_config['tokenizer_name'])
            self.max_length = self.seq2seq_config['max_length']
            self.num_beams = self.seq2seq_config['num_beams']
            self.max_summary_length = self.seq2seq_config['max_summary_length']
        else:
            self.bert_config = config['berts']
            if model_name == 'distilbert-base-uncased':
                self.dist_config = self.bert_config['distilbert_config']
                self.extractive_model = Summarizer(model_name, hidden=self.dist_config['hidden'], hidden_concat=self.dist_config['hidden_contact'])
            else:
                self.extractive_model = Summarizer(model_name)

    def __call__(self, text):
        if self.extractive_model is not None:
            summary = self.extractive_model(text)
            return summary

        inputs = self.tokenizer(text, max_length=self.max_length, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=self.num_beams, max_length=self.max_summary_length, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
