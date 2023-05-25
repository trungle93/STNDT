#!/bin/bash
# Author: Trung Le
# Scripts for downloading NLB data

mkdir data
dandi download https://dandiarchive.org/dandiset/000128 --output-dir data # replace URL with URL for dataset you want
# URLs for NLB datasets are:
# - MC_Maze: https://dandiarchive.org/dandiset/000128
# - MC_RTT: https://dandiarchive.org/dandiset/000129
# - Area2_Bump: https://dandiarchive.org/dandiset/000127
# - DMFC_RSG: https://dandiarchive.org/dandiset/000130
# - MC_Maze_Large: https://dandiarchive.org/dandiset/000138
# - MC_Maze_Medium: https://dandiarchive.org/dandiset/000139
# - MC_Maze_Small: https://dandiarchive.org/dandiset/000140