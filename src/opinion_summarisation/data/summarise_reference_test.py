import pandas as pd
import sys
import os
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from dotenv import load_dotenv
from gensim.summarization import summarize
from nltk.tokenize import sent_tokenize

# Load environment variables from .env file
load_dotenv()
load_path = os.getenv('LOCAL_ENV') + 'data/processed/summarisation_data/ref_reviews.csv'
save_path = os.getenv('LOCAL_ENV') + 'data/processed/summarisation_data/summarised_ref_reviews.csv'

# Load the CSV file
data = pd.read_csv(load_path)

text_summaries = []

for index, row in data.iterrows():
    review = row['Review Text']
    sentences = sent_tokenize(review)  # Split the review into sentences
    if len(sentences) > 0:
        summary = summarize(review, split=True, ratio=0.2)  # Set the desired summary length
        if summary:
            text_summaries.append(summary[0])
        else:
            text_summaries.append('')
    else:
        text_summaries.append('')

# Add the text summaries to the DataFrame
data['Text Summary'] = text_summaries

# Remove the rows with empty text summaries
data = data[data['Text Summary'] != '']

# Save the updated DataFrame to a new CSV file
data.to_csv(save_path, index=False)
