# Set some environment stuffs
export TRAINING_MODE='simple'

# Create and process datasets

echo "Preparing and cleaning data..."

python src/aspect_classification/data/make_dataset.py

# train aspect classifiers

python src/aspect_classification/models/train.py

# Evaluation step

echo "Making asoect label predictions on unseen data..."

python src/aspect_classification/models/evaluate.py


# train sentiment classifiers

echo "Training sentiment classifiers..."

python src/sentiment_classification/models/train.py


# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python src/sentiment_classification/models/evaluate.py


# Opinion summarisation


echo "Training the summarisation step... "

python src/sentiment_classification/models/train.py


# Opinion Summarisation evaluation 

echo "Making predictions on the summarisation... "

python src/sentiment_classification/models/evaluate.py