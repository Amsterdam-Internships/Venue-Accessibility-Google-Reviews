import pandas as pd
import os
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from dotenv import load_dotenv
from preprocessing import Preprocessor

# Load environment variables from .env file
load_dotenv()
preprocessor = Preprocessor()

def select_reviews():
    reviews = pd.read_csv(load_path)
    selected_reviews = preprocessor.remove_rows(reviews, 'JoinedReview')
    selected_reviews.to_csv(save_path, index=False)

# Example usage
load_path = os.getenv('LOCAL_ENV') + 'data/interim/grouped_reviews.csv'
save_path = os.getenv('LOCAL_ENV') + 'data/interim/selected_review_summaries.csv'

select_reviews()