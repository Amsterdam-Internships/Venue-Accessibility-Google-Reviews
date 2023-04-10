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

def gensim_formatting(df):
    formatted = {}
    df['Venues', 'Aspect', 'Sentiment']
    
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

def remove_empty_vals(df):
    df = df[df["Text"].notna()]
    if 'Rating' not in df:
        df = df[df["Sentiment"]!="0 stars"]
        return df
    else:
        df["Text"] = df["Text"].apply(lambda x: x.replace("\n", ' '))
        df['Rating']
        df = df.loc[df["Rating"] > 0.0, :]
        df["SentenceCount"] = df["Text"].apply(lambda x: len(nltk.sent_tokenize(x)))
        df = df[df["SentenceCount"]!=0]
        df = df.replace([np.nan, '\n'], '')
        return df


def clean_and_select(df, columns):
    if 'Aspect' in df:
        df_selected = rename_columns(df[columns], ["Review"],  ["Text"])
        df_cleaned = remove_empty_vals(df_selected)
        target_aspects = ['Toilets', 'Transport & Parking', 'Wheelchair']
        selected_aspects = filter_aspects(df_cleaned, target_aspects)
        selected_aspects["Venue"] = selected_aspects["Venue"].apply(lambda x: get_venue_name(x))
        return convert_rating(selected_aspects)
    else:
        df_selected = rename_columns(df[columns], ["Review Text", "Review Rate"],["Text","Sentiment"])
        cleaned_df = remove_empty_vals(df_selected)
        cleaned_sentiments = clean_sentiment(cleaned_df)
        return  convert_rating(cleaned_sentiments)
    
def replace_substrings(s):
    # Define a dictionary of substring replacements
    replacements = {' stars': '', ' star':'', ' stars ':''}
    # Loop through the dictionary and use re.sub to replace each substring
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


def pick_sentiment(x):
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
        df["Sentiment"] = df["Sentiment"].apply(lambda x : pick_sentiment(x))
    else:
        df["Sentiment"] = df["Rating"].apply(lambda x : pick_sentiment(x))
    df['Label'] = df["Sentiment"].map({'positive': 1, 'negative': 0})
    df = df[df['Sentiment'] != 'neutral']
    return df


def tokenize_text(df):
    rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    df["Text"] = df["Text"].apply(lambda x: nltk.regexp_tokenize(x, rule))
    return df


