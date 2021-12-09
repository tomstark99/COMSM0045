#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --account comsm0045
#SBATCH --reservation=comsm0045-coursework
#SBATCH --time 0-01:00
#SBATCH --mem 80GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cnn.py --epoch 400 --batch-size 64 --learning-rate 1e-3 --use-cuda --full-train 

