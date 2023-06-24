import re
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
lemmatizer = WordNetLemmatizer()

class Preprocessor(object):
    def __init__(self, ):
        self.stopwords_list =  set(nltk.corpus.stopwords.words('english'))
        self.regex_patterns = {"punctuation" : r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", "google tags": r'\(Translated by Google\)|\(Original\)'}
        self.tokenizer = {'regex':nltk.RegexpTokenizer(self.regex_patterns['punctuation']), 'sent': nltk.sent_tokenize(self.regex_patterns['punctuation'])}
        self.lemmatizer = WordNetLemmatizer()
        self.labels_map = {
        re.compile(r'^(?i)\bother\b'): "Overview",
        re.compile(r'(?i)\b(transport|parking)\b'): "Transport & Parking",
        re.compile(r'^(?i)\bentrance\b|^(?i)general\saccessibility|general\saccess$|wheelchair|wheechair|^(?i)noise\slevels$'): "Access",
        re.compile(r'^(?i)\btoilets\b|^(?i)\btoilet\b|^(?i)\btoielts\b'): "Toilets",
        re.compile(r'^(?i)\bstaff\b'): "Staff",
        re.compile(r'^(?i)\bpositive\b'): "Positive",
        re.compile(r'^(?i)\bnegative\b'): "Negative",
        re.compile(r'^(?i)\bneutral\b|(?i)\bnetural\b'): "Neutral"
        }

    def remove_columns(self):
        pass
    
    def explode_rows(self, df, column):
        """
        This function explodes the rows in the column that contains comma-separated values.

        Args:
            df (pandas.DataFrame): The DataFrame containing the 'Aspect' column.

        Returns:
            pandas.DataFrame: The DataFrame with exploded rows.
        """
        df[column] = df[column].apply(lambda x: x.split(', ') if isinstance(x, str) and ', ' in x else x)
        df = df.explode(column)
        df[column] = df[column].str.strip()
        return df


    def relabel(self, df, columns):
        """
        This function should take the Aspect column and relabel the aspects to the Euan's Guide format.

        Args:
            df (pandas.DataFrame): The DataFrame containing the gold labels column.

        Returns:
            pandas.DataFrame: The DataFrame with relabeled euans guide formatted labels.
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
        if columns[1] == "Aspect":
            return self.explode_rows(df, columns[1])
        else:
            return df

    
    def remove_rows(self, df):
        df = df.dropna(subset=['Gold Aspect Labels', 'Gold Sentiment Labels'])
        df = df[df['Gold Aspect Labels'] != '']
        df = df[df['Gold Sentiment Labels'] != '']
        return df
    def remove_stopwords(self):
        pass
    def tokenize(self):
        pass
    def vectorize(self):
        pass
    def split_aspects(self, column):
        column['Split Aspect Labels'] = column['Gold Aspect Labels'].apply(lambda x: x.split(", "))
        column['Aspect'] = column.explode('Split Aspect Labels')
        return column