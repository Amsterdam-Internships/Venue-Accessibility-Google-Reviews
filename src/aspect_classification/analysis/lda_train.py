import gensim
from gensim import models
import os
import sys
import pathlib
import pandas as pd
from data.data_cleaning import gensim_processing
from dotenv import load_dotenv


def process_data(data):
    tokenised_euans_reviews = data['Text'].apply(gensim_processing)
    dictionary = gensim.corpora.Dictionary(tokenised_euans_reviews)
    euans_corpus = [dictionary.doc2bow(review) for review in tokenised_euans_reviews]
    return euans_corpus, dictionary


def train_model(corpus, dictionary):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # Train an LDA model on the corpus
    tfidf_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                                id2word=dictionary,
                                                num_topics=10,
                                                random_state=42,
                                                passes=10)
    return tfidf_lda_model
    

def save_model(model):
    # Load environment variables from .env file
    load_dotenv()
    # Get current directory
    current_dir = os.getcwd()
    # Get parent directory
    parent_dir = os.path.join(current_dir, '..')
    # Append parent directory to sys.path
    sys.path.append(parent_dir)
    cwd = pathlib.Path.cwd().parent
    model.save(cwd.joinpath('/models/'))
    


if __name__ == '__main__':
    processed_train_path = os.environ.get('PROCESSED_TRAIN_DATA')
    euans_data = pd.read_csv(processed_train_path)
    data, dict = process_data(euans_data)
    trained_model = train_model(data, dict)
    save_model(trained_model)
    print('The TFIDF-LDA model is done training!')