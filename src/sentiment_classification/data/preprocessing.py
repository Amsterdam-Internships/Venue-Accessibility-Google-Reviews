import nltk
import re
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
        re.compile(r'^(?i)\bpositive\b|\bnegative\b'): "mixed"
        }
    

    
    def encode_datasets(self, text):
        new_encodings = my_pipeline.tokenizer(text, truncation=True, padding=True, max_length=512)
        return new_encodings
    
    def remove_columns(self):
        pass

    def relabel(self, df, columns):
        """
        This function should take the Sentiment column and relabel the Sentiment to the needed Guide format.

        Args:
            df (pandas.DataFrame): The DataFrame containing the gold labels column.

        Returns:
            pandas.DataFrame: The DataFrame with re-formatted labels.
        """
        gold_labels = df[columns[0]].values.tolist()
        for i, label in enumerate(gold_labels):
            for pattern, euans_label in self.labels_map.items():
                labels = re.split(r'\s*,\s*', label.strip())  # Split labels on comma and whitespace
                if len(labels) > 1:
                    for single_label in labels:
                        if pattern.match(single_label):
                            gold_labels[i] = gold_labels[i].replace(single_label, euans_label)
                else:
                    if pattern.match(label):
                        gold_labels[i] = euans_label
        df[columns[1]] = gold_labels
    
    def remove_rows(self, df, column):
        df = df[df[column] != 'neutral']
        return df
    def remove_stopwords(self):
        pass
    def tokenize(self):
        pass
    def vectorize(self):
        pass
 
    def create_datasets(self, data):
        labels = my_pipeline.encode_labels(data['Sentiment'].values.tolist())
        texts = data['Sentences'].values.tolist()
        print(len(labels), len(texts))
        encodings = self.encode_datasets(texts)
        new_dataset = EuansDataset(encodings, labels)
        return new_dataset
    