#!/bin/bash
python -u ray_random.py --exp-config configs/$1.yaml --samples 120
python -u scripts/nlb_ensemble.py $1