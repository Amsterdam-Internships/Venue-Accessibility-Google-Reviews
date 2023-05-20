# Set some environment stuffs
export TRAINING_MODE='simple'
source /Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/.env

# Create and process datasets

echo "Preparing and cleaning data..."

python "${LOCAL_ENV}src/aspect_classification/data/make_dataset.py"


# train aspect classifiers

python "${LOCAL_ENV}src/aspect_classification/models/train.py"

# Evaluation step

echo "Making asoect label predictions on unseen data..."

python "${LOCAL_ENV}src/aspect_classification/models/evaluate.py"

# Create graphs of evaluation metrics 

echo "Creating graphs of evaluation metrics..."

python "${LOCAL_ENV}src/aspect_classification/models/visualisations.py"


# train sentiment classifiers

echo "Training sentiment classifiers..."

python "${LOCAL_ENV}src/sentiment_classification/models/train.py"


# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python "${LOCAL_ENV}src/sentiment_classification/models/evaluate.py"


# Opinion summarisation


echo "Training the summarisation step... "

python "${LOCAL_ENV}src/sentiment_classification/models/train.py"


# Opinion Summarisation evaluation 

echo "Making predictions on the summarisation... "

python "${LOCAL_ENV}src/sentiment_classification/models/evaluate.py"