# Set some environment stuffs
export TRAINING_MODE='simple'
source /Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/.env

# Create and process datasets

echo "Preparing and cleaning data..."

python "${LOCAL_ENV}src/aspect_classification/data/make_dataset.py"


# train aspect classifiers

python "${LOCAL_ENV}src/aspect_classification/models/train.py"

# Evaluation step

echo "Making aspect label predictions on unseen data..."

python "${LOCAL_ENV}src/aspect_classification/models/evaluate.py"

# Create graphs of evaluation metrics 

echo "Creating graphs of aspect evaluation metrics..."

python "${LOCAL_ENV}src/aspect_classification/models/visualisations.py"


# train sentiment classifiers

echo "Training sentiment classifiers..."

python "${LOCAL_ENV}src/sentiment_classification/models/train.py"


# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python "${LOCAL_ENV}src/sentiment_classification/models/evaluate.py"


echo "Creating graphs of sentiment evaluation metrics..."

python "${LOCAL_ENV}src/sentiment_classification/models/visualisations.py"

# Grouping reviews

echo "Grouping review sentences by aspect..."

python "${LOCAL_ENV}src/opinion_summarisation/data/group_test_reviews.py"

# selecting longer reviews

echo "Selecting longer reviews to summarise..."

python "${LOCAL_ENV}src/opinion_summarisation/data/select_summary_reviews.py"

# Opinion summarisation training


echo "Training the summarisation step... "

python "${LOCAL_ENV}src/opinion_summarisation/models/train.py"


# Opinion Summarisation evaluation 

echo "Making predictions on the summarisation... "

python "${LOCAL_ENV}src/opinion_summarisation/models/evaluate.py"


echo "Creating graphs for summarisation evaluation... "

python "${LOCAL_ENV}src/opinion_summarisation/models/visulisations.py"