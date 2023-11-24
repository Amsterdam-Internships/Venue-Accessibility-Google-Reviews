#!/bin/bash

#SBATCH --job-name=SC-Baseline
#SBATCH --time=72:15:00
 #SBATCH --nodes=1
 #SBATCH --ntasks-per-node=1
 #SBATCH --partition=knlq   
# Load any necessary modules
module load cuda11.1/toolkit/11.1.1
module load cuDNN/cuda11.1/8.0.5

conda init bash

export LOCAL_ENV=/var/scratch/mbn781/Venue-Accessibility-Google-Reviews/

echo "LOCAL_ENV is set to: $LOCAL_ENV"

source ~/.bashrc
# Activate your desired Python environment, if needed
conda activate /var/scratch/mbn781/anaconda3/envs/BachelorsProject

# Explicitly set CUDA-related environment variables after activating conda environment
export PATH=/path/to/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/path/to/cuda-11.1/lib64:$LD_LIBRARY_PATH

# Change to your working directory
cd /var/scratch/mbn781/Venue-Accessibility-Google-Reviews

# Test that the GPU is working

echo "Is the GPU being used..."

python "${LOCAL_ENV}scripts/gpu_test.py"


# Set some environment stuffs
export TRAINING_MODE='simple'
source ./.env

# # Create and process datasets

# echo "Preparing and cleaning data..."

# python "${LOCAL_ENV}/src/aspect_classification/data/make_dataset.py"

# # train aspect classifiers

# python "${LOCAL_ENV}/src/aspect_classification/models/train.py"

# Check if the GPU is actually being used

nvidia-smi

# # Evaluation step

# echo "Making aspect label predictions on unseen data..."

# python "${LOCAL_ENV}/src/aspect_classification/models/evaluate.py"

# # Create graphs of evaluation metrics 

# echo "Creating graphs of aspect evaluation metrics..."

# python "${LOCAL_ENV}/src/aspect_classification/models/visualisations.py"


# Preparing sentiment label format

# echo "Preparing sentiment label format..."

# python "${LOCAL_ENV}/src/sentiment_classification/data/make_dataset.py"

# train sentiment classifiers

echo "Training sentiment classifiers..."

python "${LOCAL_ENV}/src/sentiment_classification/models/train.py"

nvidia-smi

# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python "${LOCAL_ENV}/src/sentiment_classification/models/evaluate.py"


echo "Creating graphs of sentiment evaluation metrics..."

python "${LOCAL_ENV}/src/sentiment_classification/models/visualisations.py"

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