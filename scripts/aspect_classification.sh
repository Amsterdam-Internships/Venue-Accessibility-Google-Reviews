# Set some environment stuffs
export TRAINING_MODE='simple'

# Create and process datasets

echo "Preparing and cleaning data..."

python src/aspect_classification/data/make_dataset.py


# Feature extraction

echo "Extracting features from the cleaned data.."

python src/aspect_classification/features/build_features.py

echo "Starting training..."

# train classifiers

python src/aspect_classification/models/train.py

# Evaluation step

echo "Making predictions on unseen data..."

python src/aspect_classification/models/predict.py


# Creating evaluation reports

echo "Generating visualisations and reports..."

python src/aspect_classification/visualisation/visualise.py


