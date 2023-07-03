import sys
import pandas as pd
import re
import numpy as np
import nltk
import gensim
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
import sys
import os
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from aspect_classification.models.newpipelines import AspectClassificationPipeline 
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
lemmatizer = WordNetLemmatizer()

pipline = AspectClassificationPipeline(pipeline_type='transformer', model_type='huawei-noah/TinyBERT_General_4L_312D')
    
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
        target_aspects = ['Toilets','Transport & Parking','Access','Overview','Staff']
        df_aspects = select_aspects(target_aspects, df_cleaned)
        added_sentiments = convert_rating(df_aspects)
        cleaned_sentiments = clean_sentiment(added_sentiments)
        return cleaned_sentiments
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
    return df
    
def filter_aspects(df, aspects):
    cleaned_aspects = [aspect.strip() for aspect in aspects]
    filtered_data = df[df["Aspect"].isin(cleaned_aspects)]
    selected_aspects = filtered_data[filtered_data["Aspect"].isin(cleaned_aspects)]
    return selected_aspects




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
    df = df[df["Sentiment"] != 'neutral'].copy()
    return df

def create_sentiment(df):
    conv_df = convert_rating(df)
    labelled_df = asign_label(conv_df)
    # binary_df = remove_sentiment(labelled_df)
    return labelled_df


def tokenize_text(df):
    rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    df["Text"] = df["Text"].apply(lambda x: nltk.regexp_tokenize(x, rule))
    return df

def bert_processing(reviews):
    # Load the BERT tokenizer
    tokenizer = pipline.tokenizer(pipline.model_name)

    preprocessed_reviews = []
    for review in reviews:
        # Tokenize the text using the BERT tokenizer
        tokens = tokenizer.tokenize(review)
        
        # Convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        
        # Join the tokens for each review into a single string
        preprocessed_review = ' '.join(tokens)
        preprocessed_reviews.append(preprocessed_review)
    return preprocessed_reviews
    
