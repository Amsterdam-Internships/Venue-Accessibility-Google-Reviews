import yaml
import torch
import sys
import os   
from dotenv import load_dotenv
sys.path.append(os.getenv('LOCAL_ENV')+'/src')
import transformers
from summarizer import Summarizer

# Load environment variables from .env file
load_dotenv()

config_path = os.getenv('LOCAL_ENV') + '/src/opinion_summarisation/models/config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class SummarizationPipeline:
    def __init__(self, model_name, model_type=None):
        if model_type == 'seq2seq':
            self.config = config['seq2seq_config']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.config['model_name'])
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config['tokenizer_name'])
            self.max_length = self.config['max_length']
            self.num_beams = self.config['num_beams']
            self.max_summary_length = self.config['max_summary_length']
        else:
            self.bert_config = config['berts']
            if model_name == 'distilbert-base-uncased':
                self.config = self.bert_config['distilbert_config']
                self.model = self.config['model_name']
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                self.extractive_model = Summarizer(model_name, hidden=self.config['hidden'], hidden_concat=self.config['hidden_contact'])
            elif model_name == 'bert-extractive-summarizer':
                self.config = self.bert_config['bert_config']
                self.model = self.config['model_name']
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                self.extractive_model = Summarizer(model_name)  
            else:
                self.config = self.bert_config['s_bert_config']
                self.model = self.config['model_name']
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                self.extractive_model = Summarizer(model_name)
        # Update the max_length attribute
            if isinstance(self.tokenizer.model_max_length, dict):
                if 'max_len' in self.tokenizer.model_max_length:
                    self.max_length = self.tokenizer.model_max_length['max_len']
                else:
                    self.max_length = self.tokenizer.model_max_length['max_length']
            else:
                self.max_length = self.tokenizer.model_max_length
        


