import pandas as pd
import os
import sys
sys.path.append(os.getenv('LOCAL_ENV') + '/src')
from dotenv import load_dotenv
from preprocessing import Preprocessor
# Load environment variables from .env file
load_dotenv()
preprocessor = Preprocessor()

def group_reviews_by_aspect():
    split_reviews = pd.read_csv(load_path)
    expanded_reviews = preprocessor.expand_aspects(split_reviews)
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
            # Group already exists, check if the review is already in the JoinedReview
            existing_index = grouped_reviews[group_exists].index[0]
            existing_joined_review = grouped_reviews.loc[existing_index, 'JoinedReview']
            
            # Check if the review is already present in the existing JoinedReview
            if review not in existing_joined_review:
                joined_review = existing_joined_review + ' ' + review
                grouped_reviews.loc[existing_index, 'JoinedReview'] = joined_review
        else:
            # Group does not exist, create a new row for the group
            new_row = {'Venue Name': venue, 'Aspect': aspect, 'Sentiment': sentiment, 'JoinedReview': review}
            grouped_reviews = grouped_reviews.append(new_row, ignore_index=True)

    # Set the data type of the "JoinedReview" column to object
    grouped_reviews['JoinedReview'] = grouped_reviews['JoinedReview'].astype('object')

    return grouped_reviews


# Example usage
load_path = os.getenv('LOCAL_ENV') + '/data/interim/predicted_sentiment_labels.csv'
save_path = os.getenv('LOCAL_ENV') + '/data/interim/grouped_reviews.csv'
grouped_reviews = group_reviews_by_aspect()

# Remove duplicates
grouped_reviews = grouped_reviews.drop_duplicates()

# Save the grouped reviews to a CSV file
grouped_reviews.to_csv(save_path, index=False)

