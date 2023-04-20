
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import torch
import sklearn
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

import logging
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/BachelorsProject/euans_reviews.csv')

test_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/BachelorsProject/google_reviews.csv")

euans_reviews = train_df.Text.values.tolist()

import time
import json
euans_path = '/content/drive/MyDrive/Colab Notebooks/BachelorsProject/euans sumarries/euans_reviews.json'

from transformers import pipeline
from textblob import TextBlob
import random
summariser = pipeline('summarization',model='sshleifer/distilbart-cnn-12-6', device=device)

"""
We want the review text per venue, aspect and sentiment.
"""
file = open(euans_path, "a")

for reviews in euans_reviews:
  summary = summariser(reviews, max_length=50, min_length=10)
  file.write(json.dumps(summary))

  # Check if 10 minutes have passed
  if time.time() % 600 == 0:
      # Flush the file buffer to ensure data is written to disk
      file.flush()



euans_sample['target_text'] = [summary['summary_text'] for summary in summaries]
euans_sample.head()

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)

n = round(len(train_df) * 0.8)

m = round(len(train_df) * 0.2)

from sklearn.metrics.pairwise import cosine_similarity
train_df.rename(columns={'Text': 'input_text'}, inplace=True)

google_sample = random.sample(grouped_google.Text.values.tolist(), round(5102*0.2))

import random
X_train = train_df.sample(n=n, random_state=42)

X_val = train_df.sample(n=m, random_state=42)

model.train_model(X_train, eval_data=X_val)

model.save_model('/content/drive/MyDrive/Colab Notebooks/BachelorsProject/models')

validation_results = model.eval_model(X_val)

google_summaries = model.predict(google_sample)

# Commented out IPython magic to ensure Python compatibility.
# %pip install rouge

from rouge import Rouge

rouge = Rouge()

google_eval = rouge.get_scores(google_summaries, google_reviews, avg=True, ignore_empty=True, rouge_n=(1, 2))