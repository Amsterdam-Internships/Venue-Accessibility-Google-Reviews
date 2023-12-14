#!/bin/bash

#SBATCH --job-name=SC-Baseline
#SBATCH --time=72:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --constraint=TitanX   # Use RTX2080Ti GPUs
#SBATCH --gres=gpu:1 

# Load any necessary modules
module load cuda11.1/toolkit/11.1.1
module load cuDNN/cuda11.1/8.0.5

conda init bash
CUDA_LAUNCH_BLOCKING=1
export LOCAL_ENV=/var/scratch/mbn781/Venue-Accessibility-Google-Reviews/

echo "LOCAL_ENV is set to: $LOCAL_ENV"

source ~/.bashrc
# Activate your desired Python environment, if needed
conda activate /var/scratch/mbn781/anaconda3/envs/BachelorsProject
# Change to your working directory
cd /var/scratch/mbn781/Venue-Accessibility-Google-Reviews

# Test that the GPU is working

echo "Is the GPU being used..."

python "${LOCAL_ENV}scripts/gpu_test.py"


# Set some environment stuffs
export TRAINING_MODE='simple'
source ./.env

nvidia-smi

# train sentiment classifiers

echo "Training sentiment classifiers..."
# test

# python "${LOCAL_ENV}/src/sentiment_classification/models/train.py"

nvidia-smi

# Sentiment evaluation step

echo "Making sentiment label predictions on unseen data..."

python "${LOCAL_ENV}/src/sentiment_classification/models/evaluate.py"

echo "Creating graphs of sentiment evaluation metrics..."

python "${LOCAL_ENV}/src/sentiment_classification/models/visualisations.py"


