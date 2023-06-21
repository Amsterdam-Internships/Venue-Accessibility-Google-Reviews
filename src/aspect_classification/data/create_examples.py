import pandas as pd
import random
import os
from faker import Faker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

fake = Faker()

aspect_list = [
    'Toilets',
    'Transport & Parking',
    'Access',
    'Overview',
    'Staff'
]

reviews = []
for _ in range(700):
    venue_name = fake.company()
    review = fake.sentence()
    rate = random.randint(1, 5)
    review_time = fake.date_between(start_date='-1y', end_date='today')
    review_text = fake.paragraph()
    aspect = random.choice(aspect_list)
    sentiment = random.choice(['Positive', 'Negative'])  # Add a dummy sentiment value

    reviews.append([venue_name, review, rate, review_time, review_text, aspect, sentiment])  # Include sentiment in the list

df = pd.DataFrame(reviews, columns=['Venue Name', 'Review', 'Rate', 'Review Time', 'Review Text', 'Aspects', 'Sentiment'])  # Include sentiment column
file_path = os.getenv('LOCAL_ENV')
df.to_csv(file_path + 'data/processed/aspect_classification_data/test_example.csv', index=False)
