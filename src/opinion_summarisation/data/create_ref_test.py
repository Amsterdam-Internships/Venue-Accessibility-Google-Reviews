import random
import os
import sys
import csv
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from faker import Faker
from dotenv import load_dotenv

load_dotenv()  # Call the function to load environment variables

fake = Faker()

# Set the number of reviews to generate
num_reviews = 700

# Create a list to store the generated reviews
reviews = []

# Generate reviews for restaurants
for _ in range(num_reviews):
    venue_name = fake.company()
    aspect = random.choice(['Overview', 'Transport & Parking', 'Access', 'Toilets', 'Staff'])
    sentiment = random.choice(['Positive', 'Negative'])
    review_text = fake.paragraphs(nb=10)  # Generate a review with 10 paragraphs
    
    review = {
        'Venue Name': venue_name,
        'Aspect': aspect,
        'Sentiment': sentiment,
        'Review Text': '\n\n'.join(review_text)  # Combine the paragraphs into a single review text
    }
    
    reviews.append(review)

# Generate reviews for museums
for _ in range(num_reviews):
    venue_name = fake.company_suffix() + ' Museum'
    aspect = random.choice(['Overview', 'Transport & Parking', 'Access', 'Toilets', 'Staff'])
    sentiment = random.choice(['Positive', 'Negative'])
    review_text = fake.paragraphs(nb=10)  # Generate a review with 10 paragraphs
    
    review = {
        'Venue Name': venue_name,
        'Aspect': aspect,
        'Sentiment': sentiment,
        'Review Text': '\n\n'.join(review_text)  # Combine the paragraphs into a single review text
    }
    
    reviews.append(review)

# Define the CSV file path
csv_file = os.getenv('LOCAL_ENV') + 'data/processed/summarisation_data/ref_reviews.csv'

# Write the reviews to the CSV file
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['Venue Name', 'Aspect', 'Sentiment', 'Review Text']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(reviews)

print(f'Reviews saved to {csv_file}.')
