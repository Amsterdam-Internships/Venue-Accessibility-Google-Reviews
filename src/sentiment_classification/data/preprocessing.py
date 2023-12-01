import nltk
import re
import os
import sys
import numpy as np
import ast
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sentiment_classification.models.sentiment_pipeline import EuansDataset, SentimentClassificationPipeline

my_pipeline = SentimentClassificationPipeline(pipeline_type='transformer')

class Preprocessor(object):
    def __init__(self):
        self.stopwords_list = set(stopwords.words('english'))
        self.regex_patterns = {
            "punctuation": r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",
            "google tags": r'\(Translated by Google\)|\(Original\)'
        }
        self.tokenizer = {
            'regex': RegexpTokenizer(self.regex_patterns['punctuation']),
            'sent': nltk.sent_tokenize(self.regex_patterns['punctuation'])
        }
        self.lemmatizer = WordNetLemmatizer()
        self.labels_map = {
        re.compile(r'^(?i)\bpositive\b'): "positive",
        re.compile(r'^(?i)\bnegative\b'): "negative",
        re.compile(r'^(?i)\bneutral\b|(?i)\bnetural\b'): "neutral",
        re.compile(r'^(?i)\bpositive, negative\b|(?i)\bnegative, positive\b'): "mixed"
        }
    

    
    def encode_datasets(self, text):
        new_encodings = my_pipeline.tokenizer(text, truncation=True, padding=True, max_length=512)
        return new_encodings
    
    def remove_columns(self):
        pass
    

    def relabel(self, df, column):
        """
        This function should take the Sentiment column and relabel the Sentiment to the needed Guide format.

        Args:
            df (pandas.DataFrame): The DataFrame containing the gold labels column.

        Returns:
            pandas.DataFrame: The DataFrame with re-formatted labels.
        """
        gold_labels = df[column].values.tolist()
        for i, label in enumerate(gold_labels):
            for pattern, euans_label in self.labels_map.items():
                if isinstance(label, str) and pattern.match(label):
                    gold_labels[i] = euans_label

        df['Sentiment'] = gold_labels
        return df

    
    
    def remove_rows(self, df, column):
        df = df.dropna(subset=[column])
        df = df[df[column].str.strip() != '']
        df = df[df[column] != 'mixed']
        return df

    def remove_stopwords(self):
        pass
    def convert_to_list(self, data):
        labels = data.Sentiment.values.tolist()
        reviews = data.Sentences.values.tolist()
        return labels, reviews
    
    def split_data(self, data):
        condition = data['Sentiment'].apply(lambda x: len(x) > 1)
        data['has list'] = data['Sentiment'].where(condition)
        data = data.explode('has list')
        labels, reviews = self.convert_to_list(data)
        # Get unique labels using set
        labels = my_pipeline.label_binarizer.fit_transform(labels)
        # labels = labels.astype(np.float32)
        return labels,reviews
    
    def split_sentiments(self, column, df):
        df[column] = df[column].apply(lambda x: self.flatten_list(x))
        df['Sentiment'] = column.explode('Split Sentiment Labels')
        return column
 
    def create_datasets(self, data):
        labels, texts = self.split_data(data)
        encodings = self.encode_datasets(texts)
        new_dataset = EuansDataset(encodings, labels)
        return new_dataset
    