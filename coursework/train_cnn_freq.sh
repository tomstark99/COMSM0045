#!/usr/bin/env bash
#SBATCH --account comsm0045
#SBATCH --reservation=comsm0045-coursework
#SBATCH --partition gpu
#SBATCH --time 0-01:30
#SBATCH --mem 100GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cnn.py --batch-size 64 --learning-rate 1e-3 --use-cuda --full-train --epochs 400 --val-frequency 1 --freq-mask


