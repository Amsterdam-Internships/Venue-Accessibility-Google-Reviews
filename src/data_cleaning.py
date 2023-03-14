import pandas as pd
import re
import nltk


def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def rename_and_reformat(df):
    df = df.rename(columns={"Review": "Text"})
    df['Text'] = df['Text'].astype(str)
    return df

def clean_train_df(df):
    wanted_columns = drop_train(df)
    selected_aspects = filter_aspects(wanted_columns,'Toilets', 'Transport & Parking')
    selected_aspects["Venue"] = selected_aspects["Venue"].apply(lambda x: get_venue_name(x))
    return  rating_to_sent(selected_aspects)

def clean_sentiment(df):
    to_remove = [" stars", " tar"]
    df["Sentiment"] = df["Sentiment"].map(lambda x: re.sub("|".join(to_remove), "", x))
    return df

def clean_test_df(df):
    wanted_columns = drop_test(df)
    tokenised_data = tokenize_text(df)
    return tokenised_data
    clean_sent = clean_sentiment(tokenised_data)
    return  rating_to_sent(clean_sent)
  

def drop_train(df):
    df = df.drop(columns=["City", "Country"])
    df = rename_and_reformat(df)
    df["SentenceCount"] = df["Text"].apply(lambda x: count_sentences(x))
    df = df[df["Text"].notna()]
    return df
## merge common parts of code together 
def drop_test(df):
    df = df.drop(columns=["Review Time"])
    df = df.rename(columns={"Review Text": "Text", "Review Rate": "Sentiment"})
    return df


## This needs to change for the test set
def filter_aspects(df, first, second):
    target_aspects = [first, second]
    filtered_data = df[df["Aspect"].isin(target_aspects)]
    return filtered_data


def pick_sentiment(x):
    if x > 4.0:
        return 'positive'
    elif x < 3.0:
        return 'negative'
    else:
        return 'neutral'


def get_venue_name(venue):
     return ' '.join(venue.split('|')[4].split("-")[:-1])

def rating_to_sent(df):
    df["Sentiment"] = df["Rating"].apply(lambda x : pick_sentiment(x))
    df['Label'] = df["Sentiment"].map({'positive': 1, 'negative': 0})
    return df


def tokenize_text(df):
    rule = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    df["Text"] = df["Text"].apply(lambda x: nltk.regexp_tokenize(x, rule))
    return df


