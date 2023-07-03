import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
import sys
import os
import ast
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
import numpy as np
from aspect_classification.models.newpipelines import EuansDataset, AspectClassificationPipeline
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
lemmatizer = WordNetLemmatizer()
pipeline = AspectClassificationPipeline(pipeline_type='transformer')

class Preprocessor(object):
    def __init__(self, ):
        self.stopwords_list =  set(nltk.corpus.stopwords.words('english'))
        self.regex_patterns = {"punctuation" : r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", "google tags": r'\(Translated by Google\)|\(Original\)'}
        self.tokenizer = {'regex':nltk.RegexpTokenizer(self.regex_patterns['punctuation']), 'sent': nltk.sent_tokenize(self.regex_patterns['punctuation'])}
        self.lemmatizer = WordNetLemmatizer()
        self.labels_map = {
        re.compile(r'^(?i)\bother\b'): "Overview",
        re.compile(r'(?i)\b(transport|parking)\b'): "Transport & Parking",
        re.compile(r'^(?i)\bentrance\b|^(?i)general\saccessibility|general\saccess$|wheelchair|wheelchaiir|wheechair|^(?i)noise\slevels$'): "Access",
        re.compile(r'^(?i)\btoilets\b|^(?i)\btoilet\b|^(?i)\btoliet\b|^(?i)\btoielts\b'): "Toilets",
        re.compile(r'^(?i)\bstaff\b'): "Staff"
        }
        
    def encode_datasets(self, text):
        new_encodings = pipeline.tokenizer(text, truncation=True, padding=True, max_length=512)
        return new_encodings

    def create_datasets(self, data):
        labels, texts = self.split_data(data)
        encodings = self.encode_datasets(texts)
        new_dataset = EuansDataset(encodings, labels)
        return new_dataset
    
    def convert_to_list(self, data):
        label_strings = data.labels.values.tolist()
        labels_only = [ast.literal_eval(value) for value in label_strings]
        return [labels.split(", ") if ", " in labels else labels for labels in labels_only]


    def split_data(self, data):
        data = data.rename(columns={"Aspect": "labels", "Sentences": "text"})
        labels = self.convert_to_list(data)
        labels = pipeline.label_binarizer.fit_transform(labels)
        labels = labels.astype(np.float32)
        reviews = data.text.values.tolist()
        return labels,reviews


    def remove_columns(self):
        pass
    
    def split_and_remove_duplicates(self, df, column):
        """
        This function splits the values in a column by comma and removes duplicate values.

        Args:
            df (pandas.DataFrame): The DataFrame containing the column to be processed.
            column (str): The name of the column to split and remove duplicates.

        Returns:
            pandas.DataFrame: The DataFrame with split and deduplicated values in the specified column.
        """
        df[column] = df[column].apply(lambda x: [val.strip() for val in set(x.split(", "))] if isinstance(x, str) and (", " in x or "," in x) else x)
        df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else x)
        return df


    def relabel(self, df, columns):
        """
        This function should take the Aspect column and relabel the aspects to the Euan's Guide format.

        Args:
            df (pandas.DataFrame): The DataFrame containing the gold labels column.

        Returns:
            pandas.DataFrame: The DataFrame with relabeled euans guide formatted labels.
        """
        df[columns[0]] = df[columns[0]].astype(str).fillna('').str.strip()
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
        return self.split_and_remove_duplicates(df, columns[1])

    def remove_rows(self, df, row):
        df = df[df[row] != '']
        df['nonsense flag'] = df[row].str.contains(r'(?i)\bnonsense\b')
        df = df[df['nonsense flag'] != True]
        df = df.dropna(subset=[row])
        return df

    
    def remove_stopwords(self):
        pass
    def tokenize(self):
        pass
    def vectorize(self):
        pass
    def split_reviews(self, review):
        if len(nltk.sent_tokenize(review)) > 1:
            return nltk.sent_tokenize(review)
        else:
            return [review]
    def split_aspects(self, column):
        column['Split Aspect Labels'] = column['Gold Aspect Labels'].apply(lambda x: x.split(", "))
        # column['Aspect'] = column.explode('Split Aspect Labels')
        return column