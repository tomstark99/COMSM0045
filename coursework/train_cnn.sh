#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:10
#SBATCH --mem 16GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cnn.py
