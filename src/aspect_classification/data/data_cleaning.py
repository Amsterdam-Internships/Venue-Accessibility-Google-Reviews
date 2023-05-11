import sys
import pandas as pd
import re
import numpy as np
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
lemmatizer = WordNetLemmatizer()
    
def stem_and_lemmatize(sentence):
    return lemmatizer.lemmatize(sentence)

def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def rename_columns(df, old_cols, new_cols):
    rename_dict = {}
    for old, new in zip(old_cols, new_cols):
        rename_dict[old] = new
    # Rename columns using the dictionary
    df = df.rename(columns=rename_dict)
    return df

def removing_nans(df):
    df = df[df["Text"].notna()]
    if 'Rating' not in df:
        df = df[df["Sentiment"]!="0 stars"]
        return df
    else:
        # split this up 
        df["Text"] = df["Text"].apply(lambda x: x.replace("\n", ' '))
        df = df.loc[df["Rating"] > 0.0, :]
        df["SentenceCount"] = df["Text"].apply(lambda x: len(nltk.sent_tokenize(x)))
        df = df[df["SentenceCount"]!=0]
        df = df.replace([np.nan, '\n'], '')
        return df
    
def select_aspects(aspects, df):
    selected_aspects = filter_aspects(df, aspects)
    selected_aspects["Venue"] = selected_aspects["Venue"].apply(get_venue_name)
    return selected_aspects

def remove_translate_tags(df):
    pattern = r'\(Translated by Google\)|\(Original\)'  # Define the regular expression pattern
    df['Text'] = df['Text'].str.replace(pattern, '', regex=True)
    return df

def cleaning_selector(df, columns):
    if 'Aspect' in df:
        df_selected = rename_columns(df[columns], ["Review"],  ["Text"])
        df_cleaned = removing_nans(df_selected)
        target_aspects = ['Toilets', 'Transport & Parking',
                          'Wheelchair', 'Staff', 'Overview', 'Access']
        df_aspects = select_aspects(target_aspects, df_cleaned)
        return convert_rating(df_aspects)
    else:
        old_columns = ["Review Text", "Review Rate"]
        new_columns = ["Text","Sentiment"]
        df_selected = rename_columns(df[columns], old_columns,new_columns)
        cleaned_df = removing_nans(df_selected)
        removed_tags = remove_translate_tags(cleaned_df)
        cleaned_sentiments = clean_sentiment(removed_tags)
        
        return  convert_rating(cleaned_sentiments)
    
def replace_substrings(s):
    replacements = {' stars': '', ' star':'', ' stars ':''}
    for old, new in replacements.items():
        s = re.sub(old, new, s)
    return s
    
def clean_sentiment(df):
    df['Sentiment'] = df['Sentiment'].apply(lambda x: replace_substrings(x))
    df["Sentiment"] = df["Sentiment"].apply(lambda x: int(x))
    return df
    
def filter_aspects(df, aspects):
    filtered_data = df[df["Aspect"].isin(aspects)]
    return filtered_data


def convert_score(x):
    x = float(x)
    if x >= 4.0:
        return 'positive'
    elif x <= 3.0:
        return 'negative'
    else:
        return 'neutral'


def get_venue_name(venue):
     return ' '.join(venue.split('|')[4].split("-")[:-1])
 
def convert_rating(df):
    if 'Rating' not in df:
        df["Sentiment"] = df["Sentiment"].apply(lambda x : convert_score(x))
    else:
        df["Sentiment"] = df["Rating"].apply(lambda x : convert_score(x))
    return df

def asign_label(df):
    df['Label'] = df["Sentiment"].map({'positive': 1, 'negative': 0})
    return df

def remove_sentiment(df):
    df = df[df['Sentiment'] != 'neutral']
    return df

def create_sentiment(df):
    conv_df = convert_rating(df)
    labelled_df = asign_label(conv_df)
    binary_df = remove_sentiment(labelled_df)
    return binary_df


def tokenize_text(df):
    rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    df["Text"] = df["Text"].apply(lambda x: nltk.regexp_tokenize(x, rule))
    return df

def lemmatize_stemming(text):
    return lemmatizer.lemmatize(text)

def gensim_processing(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def bert_processing(dataframe):
    reviews = dataframe['Text'].values.tolist()
    for i, review in enumerate(reviews[:]):
        # remove punctuation, special characters, and numbers from the text data
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(review)
        
        # remove stop words from the text data.
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # convert all text to lowercase.
        tokens = [token.lower() for token in tokens]
        
        # tokenize the text data into individual words or subwords.
        tokens = nltk.word_tokenize(' '.join(tokens))
        
        reviews[i] = tokens
    return reviews
    
