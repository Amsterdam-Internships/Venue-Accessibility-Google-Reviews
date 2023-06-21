import re
import nltk
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

class Preprocessor(object):
    def __init__(self):
        self.stopwords_list =  set(nltk.corpus.stopwords.words('english'))
        self.regex_patterns = {"punctuation" : r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", "google tags": r'\(Translated by Google\)|\(Original\)'}
        self.tokenizer = {'regex':nltk.RegexpTokenizer(self.regex_patterns['punctuation'])}
        self.lemmatizer = WordNetLemmatizer()

    def count_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    def remove_rows(self, df, column_name):
        df['Sentence Count'] = df[column_name].apply(lambda x: self.count_sentences(x))
        df = df[df['Sentence Count'] >= 3]
        return df
    

