#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=00:25:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1


# Load any necessary modules
module load cuda10.0/toolkit
module load cuDNN/cuda10.0

conda init bash

source ~/.bashrc
# Activate your desired Python environment, if needed
conda activate /var/scratch/mbn781/anaconda3/envs/BachelorsProject

# Change to your working directory
cd /var/scratch/mbn781/Venue-Accessibility-Google-Reviews/scripts

# Run your command or script
# bash full_pipeline.sh
# End of script
