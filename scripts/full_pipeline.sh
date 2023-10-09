#!/bin/bash

SBATCH --job-name=baseline
SBATCH --time=03:00:00
SBATCH -N 1
SBATCH --ntasks-per-node=1
SBATCH --partition=proq
SBATCH --gres=gpu:1


# Load any necessary modules
module load cuda10.0/toolkit
module load cuDNN/cuda10.0

conda init bash

source ~/.bashrc
# Activate your desired Python environment, if needed
conda activate /var/scratch/mbn781/anaconda3/envs/BachelorsProject

# Change to your working directory
cd /var/scratch/mbn781/Venue-Accessibility-Google-Reviews

# Set some environment stuffs
export TRAINING_MODE='simple'
source ./.env

# Create and process datasets

echo "Preparing and cleaning data..."

python "${LOCAL_ENV}/src/aspect_classification/data/make_dataset.py"

# Check if the GPU is actually being used
nvidia-smi

# train aspect classifiers

python "${LOCAL_ENV}/src/aspect_classification/models/train.py"

# Evaluation step

echo "Making aspect label predictions on unseen data..."

python "${LOCAL_ENV}/src/aspect_classification/models/evaluate.py"

# Create graphs of evaluation metrics 

echo "Creating graphs of aspect evaluation metrics..."

python "${LOCAL_ENV}/src/aspect_classification/models/visualisations.py"


# # train sentiment classifiers

# echo "Training sentiment classifiers..."

# python "${LOCAL_ENV}/src/sentiment_classification/models/train.py"


# # Sentiment evaluation step

# echo "Making sentiment label predictions on unseen data..."

# python "${LOCAL_ENV}/src/sentiment_classification/models/evaluate.py"


# echo "Creating graphs of sentiment evaluation metrics..."

# python "${LOCAL_ENV}/src/sentiment_classification/models/visualisations.py"

# # Grouping reviews

# echo "Grouping review sentences by aspect..."

# python "${LOCAL_ENV}/src/opinion_summarisation/data/group_test_reviews.py"

# # Opinion summarisation training


# echo "Training the summarisation step... "

# python "${LOCAL_ENV}/src/opinion_summarisation/models/train.py"


# # Opinion Summarisation evaluation 

# echo "Making predictions on the summarisation... "

# python "${LOCAL_ENV}/src/opinion_summarisation/models/evaluate.py"


# echo "Creating graphs for summarisation evaluation... "

# python "${LOCAL_ENV}/src/opinion_summarisation/models/visulisations.py"
