import nltk
import ast
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

class Preprocessor(object):
    def __init__(self):
        self.stopwords_list =  set(nltk.corpus.stopwords.words('english'))
        self.regex_patterns = {"punctuation" : r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", "google tags": r'\(Translated by Google\)|\(Original\)'}
        self.tokenizer = {'regex':nltk.RegexpTokenizer(self.regex_patterns['punctuation'])}
        self.lemmatizer = WordNetLemmatizer()
        
    def expand_aspects(self, reviews_df):
        expanded_rows = []

        for index, row in reviews_df.iterrows():
            aspects = ast.literal_eval(row['Predicted Aspect Labels'])
            for aspect in aspects:
                expanded_row = row.copy()
                expanded_row['Predicted Aspect Labels'] = aspect
                expanded_rows.append(expanded_row)

        expanded_df = pd.DataFrame(expanded_rows, columns=reviews_df.columns)
        return expanded_df

    def count_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    def remove_rows(self, df, column_name):
        df['Sentence Count'] = df[column_name].apply(lambda x: self.count_sentences(x))
        df = df[df['Sentence Count'] >= 3]
        return df
    
    def select_reviews(self, load_path, save_path):
        reviews = pd.read_csv(load_path)
        selected_reviews = self.remove_rows(reviews, 'JoinedReview')
        selected_reviews.to_csv(save_path, index=False)
    

