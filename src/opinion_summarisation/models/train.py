
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import torch
import os
import sklearn
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from Pipelines import SummarizationPipeline
import yaml

"""
We want the review text per venue, aspect and sentiment.
"""
with open('src/opinion_summarisation/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


pipline = SummarizationPipeline(params['max_length'],params['num_beans'],params['max_summary_length'],params['model_name'])

def train_bert_models():
    pass




if __name__ == '__main__':
    # Get the file paths from environment variables
    loaded_data_path = os.getenv('LOCAL_ENV') + # not sure yet
    saved_model_path = os.getenv('LOCAL_ENV') + 'models/opinion_summarisation'