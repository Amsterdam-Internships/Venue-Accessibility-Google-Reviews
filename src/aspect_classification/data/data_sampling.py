# %%
import pandas as pd
import os
from dotenv import load_dotenv
import nltk
import re
import string
nltk.download('punkt') 
load_dotenv()
home_path = os.getenv('LOCAL_ENV')
google_reviews = pd.read_csv(home_path + 'data/processed/aspect_classification_data/processed_google_reviews.csv')


# %%
google_reviews["Sentence Count"] = google_reviews["Text"].apply(lambda x: len(nltk.sent_tokenize(x)))

# %%
google_reviews

# %% [markdown]
# split the reviews in the sentence count column that have more than one sentence.

# %%
# Custom tokenization pattern excluding certain punctuation marks
pattern = r'\b\w+\b|[' + re.escape(string.punctuation.replace('.', '')) + '](?<!\.)'

# split the google reviews 
split_google_reviews = google_reviews.copy()
split_google_reviews['Sentences'] = split_google_reviews['Text'].apply(nltk.sent_tokenize)
split_google_reviews = split_google_reviews.explode('Sentences').reset_index(drop=True)


# %%
# Count words with custom tokenization pattern
split_google_reviews['Word Count'] = split_google_reviews['Sentences'].apply(lambda x: len(nltk.regexp_tokenize(x, pattern)))

# %%
pd.set_option('display.max_colwidth', None)

# %%
# Assign unique numeric ID to each review
split_google_reviews['Review ID'] = split_google_reviews.groupby('Sentences').ngroup()

split_google_reviews = split_google_reviews[split_google_reviews['Sentence Count'] > 1]

# %%
# Filter out sentences with less than 5 words
split_google_reviews = split_google_reviews[split_google_reviews['Word Count'] >= 5]

# %%
split_google_reviews

# %%
import nltk
import spacy
from spacy.util import filter_spans
from spaczz.matcher import FuzzyMatcher
from spacy import matcher
from spacy.tokens import Doc
from nltk.corpus import wordnet
from spacy.tokens import Span


# %%
nlp = spacy.blank("en")
matcher = FuzzyMatcher(nlp.vocab)

# %% [markdown]
# Use the Fuzzy matcher from spaczz and phrase matcher to look for the synonyms related to my previous regex expressipn.

# %%
#'breakfast', 'lunch', 'dinner', 'alcohol'
word_patterns = ['food', 'drink', 'lunch', 'breakfast', 'dinner', 'alcohol', 'beer', 'wine','pancakes', 'drink', 'desserts', 'gin', 'wine', 'breakfast', 'lunch', 'pasta',
                 'vegeterian', 'vegan', 'burgers', 'pasta', 'dish', 'beer', 'pizza', 'taste',
                 'food', 'cocktail', 'coffee', 'menu', 'tasty', 'delicious', 'staff', 'host',
                 'ambience', 'atmosphere', 'cozy', 'gezellig', 'service', 'pricey', 'cheap',
                 'nice place', 'great place', 'amazing place', 'good place', 'bad place',
                 'terrible place', 'great experience', 'chicken', 'burger', 'atmosphere']


# %%
def get_synonyms(word):
    synonyms = []
    synsets = wordnet.synsets(word)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

# %%
def has_synset(word):
    synsets = wordnet.synsets(word)
    return len(synsets) > 0

# %%
def make_pattern(text):
    patterns = []
    for word in text:
        synonyms = get_synonyms(word)
        words = [word] + synonyms
        pattern = [nlp(word) for word in words]
        patterns.extend(pattern)
    return patterns


# %%
new_pattern = make_pattern(word_patterns)

# %%
print(new_pattern)

# %%
fuzzy_matcher = FuzzyMatcher(nlp.vocab)
fuzzy_matcher.add("FOOD_PATTERN", new_pattern)

# %%
@spacy.Language.component("filter_noisy_tokens")
def filter_noisy_tokens(doc):
    noisy_tokens = []
    matches = fuzzy_matcher(doc)
    spans = [Span(doc, start, end) for _, start, end, _, _ in matches]
    filtered_spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in filtered_spans:
            retokenizer.merge(span)

    for span in filtered_spans:
        noisy_tokens.extend(range(span.start, span.end))

    words = [token.text for token in doc if token.i not in noisy_tokens]
    doc = Doc(doc.vocab, words=words)

    return doc



# %%
# Register max_token_index as an extension
Doc.set_extension("max_token_index", default=-1, force=True)

# %%
nlp.add_pipe("filter_noisy_tokens", last=True)



# %%
split_google_reviews

# %%
def remove_noise(google_reviews):
    noisy_reviews = []

    for _, row in google_reviews.iterrows():
        text = row['Sentences']
        doc = nlp(text)
        
        # Check if any matches were detected in the document
        if len(fuzzy_matcher(doc)) > 0:
            noisy_reviews.append(row)

    noisy_google_reviews = pd.DataFrame(noisy_reviews)
    return google_reviews.drop(noisy_google_reviews.index)

# Drop noisy rows from original dataframe
clean_split_google_reviews = remove_noise(split_google_reviews[:10000])



# %%
clean_split_google_reviews

# %%
aspects_patterns = ['width','space','entrance','Wheelchair', 'Access', 'Staff', 'Toilets', 'Transport & Parking', 'Overview', 'Noise levels']

synonyms = {word: get_synonyms(word) for word in aspects_patterns}

# Create a set of all your keywords and their synonyms for efficient lookup
keywords = set(aspects_patterns)
for word, synonym_list in synonyms.items():
    for synonym in synonym_list:
        keywords.add(synonym)

# Define a function to check if a review contains any keyword
def contains_keyword(review):
    return any(keyword in review for keyword in keywords)

# Create patterns with both the word and its synonyms
synonym_aspect_patterns = [nlp(word) for word in keywords]

print(synonym_aspect_patterns)
# Add patterns to the fuzzy matcher
fuzzy_matcher.add("RELEVANT_PATTERN", synonym_aspect_patterns)


# %%
sentences = clean_split_google_reviews['Sentences'].tolist()
batch_size = 4739  # Adjust the batch size as needed

real_relevant_reviews = []
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    docs = [nlp(text) for text in batch]
    for j, doc in enumerate(docs):
        matches = fuzzy_matcher(doc)
        if len(matches) > 0:
            row_index = i + j  # Calculate the index of the current doc in the original dataframe
            row = split_google_reviews.iloc[row_index]
            real_relevant_reviews.append(row)

real_relevant_split_google_reviews = pd.DataFrame(real_relevant_reviews)



# %%
small_split_google_reviews = split_google_reviews[:10000]

# %%
def contains_keyword(review):
    return any(keyword in review for keyword in keywords)

small_split_google_reviews['relevant'] = small_split_google_reviews['Sentences'].apply(contains_keyword)


# %%
len(small_split_google_reviews)

# %%
clean_split_google_reviews['relevant'] = clean_split_google_reviews['Sentences'].apply(contains_keyword)

# %%
clean_split_google_reviews

# %%
true_count = small_split_google_reviews['relevant'].value_counts()[True]

# %%
false_count = small_split_google_reviews['relevant'].value_counts()[False]

# %%
false_count_df = pd.DataFrame({'False Count': [false_count]})

# %%
print(true_count, false_count)

# %%
clean_true_count = clean_split_google_reviews['relevant'].value_counts()[True]
clean_false_count = clean_split_google_reviews['relevant'].value_counts()[False]

# %%
print(clean_true_count, clean_false_count)

# %%
real_relevant_split_google_reviews.to_excel(home_path+'data/processed/aspect_classification_data/new_sample2.xlsx')

# %%
# Filter reviews to only include those that contain a keyword
relevant_reviews = []
for _, row in split_google_reviews.iterrows():
    if contains_keyword(row['Sentences']):
        relevant_reviews.append(row)
relevant_split_google_reviews = pd.DataFrame(relevant_reviews)

# %%
relevant_split_google_reviews['relevant'] = relevant_split_google_reviews['Sentences'].apply(contains_keyword)

# %%
%pip install top2vec[sentence_encoders]

# %%
from top2vec import Top2Vec
documents = relevant_split_google_reviews['Sentences'].to_list()
model = Top2Vec(documents)

# %%
topics = model.get_topics(num_topics=10)

# %%
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["access"], num_topics=5)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)

# %%
print(topic_scores)

# %%
relevant_split_google_reviews

# %%
3000/len(aspects_patterns)

# %%
count = relevant_split_google_reviews[relevant_split_google_reviews['Sentences'].str.contains('wheelchair', case=False)].shape[0]

# %%
print(count)

# %%
word_counts = {}

for word in aspects_patterns:
    count = relevant_split_google_reviews[relevant_split_google_reviews['Sentences'].str.contains(word, case=False)].shape[0]
    word_counts[word] = count

# %%
print(word_counts)

# %%
selected_rows = pd.DataFrame()
target_count = 199  # Number of rows to select from each group
total_selected_count = 0  # Total count of selected rows

for word, count in word_counts.items():
    if count >= target_count:
        group_rows = relevant_split_google_reviews[relevant_split_google_reviews['Sentences'].str.contains(word, case=False)]
        selected_group_rows = group_rows.sample(target_count, random_state=42)
        selected_rows = pd.concat([selected_rows, selected_group_rows])
        total_selected_count += target_count

    # Break the loop if the desired total count is reached
    if total_selected_count >= 4834:
        break

# Reset the index of the selected rows dataframe
selected_rows = selected_rows.reset_index(drop=True)



# %%
# Assuming 'df' is your dataframe with a column 'Relevance' containing 'True' or 'False' values
relevant_rows = split_google_reviews[split_google_reviews['relevant'] == True]
irrelevant_rows = split_google_reviews[split_google_reviews['relevant'] == False]

# Calculate the number of relevant and irrelevant rows to select
num_relevant_rows = int(4839 * 0.6)  # Approximately 60% of the total desired count
num_irrelevant_rows = 4839 - num_relevant_rows

# Randomly sample the desired number of relevant and irrelevant rows
selected_relevant_rows = relevant_rows.sample(num_relevant_rows, random_state=42)
selected_irrelevant_rows = irrelevant_rows.sample(num_irrelevant_rows, random_state=42)

# Concatenate the selected rows
selected_rows = pd.concat([selected_relevant_rows, selected_irrelevant_rows])

# Reset the index of the selected rows dataframe
selected_rows = selected_rows.reset_index(drop=True)

# %%
len(selected_rows)

# %%
selected_rows.to_excel(home_path+'data/processed/aspect_classification_data/selected_rows.xlsx')

# %%



