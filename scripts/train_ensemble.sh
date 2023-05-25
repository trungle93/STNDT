#!/bin/bash
# Author: Trung Le
# Scripts for launching hyperparameter tuning and ensembling the models

python -u src/ray_random.py --exp-config configs/$1.yaml --samples 120
python -u src/nlb_ensemble.py $1