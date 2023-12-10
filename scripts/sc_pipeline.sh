#!/bin/bash

#SBATCH --job-name=SC-Baseline
#SBATCH --time=72:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=knlq   

conda init bash

export LOCAL_ENV=/var/scratch/mbn781/Venue-Accessibility-Google-Reviews/

echo "LOCAL_ENV is set to: $LOCAL_ENV"

source ~/.bashrc
# Activate your desired Python environment, if needed
conda activate /var/scratch/mbn781/anaconda3/envs/BachelorsProject
# Change to your working directory
cd /var/scratch/mbn781/Venue-Accessibility-Google-Reviews


# Set some environment stuffs
export TRAINING_MODE='simple'
source ./.env


# train sentiment classifiers

echo "Training sentiment classifiers..."

#python "${LOCAL_ENV}/src/sentiment_classification/models/train.py"

# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python "${LOCAL_ENV}/src/sentiment_classification/models/evaluate.py"

echo "Creating graphs of sentiment evaluation metrics..."

python "${LOCAL_ENV}/src/sentiment_classification/models/visualisations.py"


