#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-01:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cnn.py --learning-rate 5e-3
python train_cnn.py --learning-rate 1e-3
python train_cnn.py --learning-rate 5e-4
python train_cnn.py --learning-rate 1e-4
python train_cnn.py --learning-rate 5e-5
python train_cnn.py --learning-rate 5e-6
python train_cnn.py --learning-rate 1e-5
python train_cnn.py --learning-rate 1e-6
