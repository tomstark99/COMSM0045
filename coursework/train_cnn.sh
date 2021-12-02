#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-01:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cnn.py --epoch 200 --batch-size 32 --learning-rate 1e-3 --use-cuda --full-train --data-aug-hflip

