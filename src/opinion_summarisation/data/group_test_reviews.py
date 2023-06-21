import pandas as pd
import os
import ast
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def expand_aspects(reviews_df):
    expanded_rows = []

    for index, row in reviews_df.iterrows():
        aspects = ast.literal_eval(row['Predicted Aspect Labels'])
        for aspect in aspects:
            expanded_row = row.copy()
            expanded_row['Predicted Aspect Labels'] = aspect
            expanded_rows.append(expanded_row)

    expanded_df = pd.DataFrame(expanded_rows, columns=reviews_df.columns)
    return expanded_df

def group_reviews_by_aspect():
    split_reviews = pd.read_csv(load_path)
    expanded_reviews = expand_aspects(split_reviews)
    relevant_columns = expanded_reviews[['Venue Name', 'Sentences', 'Predicted Aspect Labels', 'Predicted Sentiment Labels']]

    # Create a new DataFrame to store the grouped reviews
    grouped_reviews = pd.DataFrame(columns=['Venue Name', 'Aspect', 'Sentiment', 'JoinedReview'])

    # Iterate over each row in the relevant columns DataFrame
    for index, row in relevant_columns.iterrows():
        venue = row['Venue Name']
        review = row['Sentences']
        aspect = row['Predicted Aspect Labels']
        sentiment = row['Predicted Sentiment Labels']

        # Check if a group with the same venue, aspect, and sentiment already exists in the grouped_reviews DataFrame
        group_exists = (grouped_reviews['Venue Name'] == venue) & (grouped_reviews['Aspect'] == aspect) & (grouped_reviews['Sentiment'] == sentiment)

        if group_exists.any():
            # Group already exists, join the review string with the existing JoinedReview in that group
            existing_index = grouped_reviews[group_exists].index[0]
            joined_review = grouped_reviews.loc[existing_index, 'JoinedReview'] + ' ' + review
            grouped_reviews.loc[existing_index, 'JoinedReview'] = joined_review
        else:
            # Group does not exist, create a new row for the group
            new_row = {'Venue Name': venue, 'Aspect': aspect, 'Sentiment': sentiment, 'JoinedReview': review}
            grouped_reviews = grouped_reviews.append(new_row, ignore_index=True)

    # Set the data type of the "JoinedReview" column to object
    grouped_reviews['JoinedReview'] = grouped_reviews['JoinedReview'].astype('object')

    return grouped_reviews


# Example usage
load_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_sentiment_labels.csv'
save_path = os.getenv('LOCAL_ENV') + 'data/interim/grouped_reviews.csv'
grouped_reviews = group_reviews_by_aspect()

# Remove duplicates
grouped_reviews = grouped_reviews.drop_duplicates()

# Save the grouped reviews to a CSV file
grouped_reviews.to_csv(save_path, index=False)

