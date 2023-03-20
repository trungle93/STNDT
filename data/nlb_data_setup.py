# curated from `basic_example.ipynb` in `nlb_tools`

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5, combine_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import argparse
import logging
logging.basicConfig(level=logging.INFO)

## If necessary, download datasets from DANDI and put them in './data/'
# !pip install dandi
# !dandi download https://dandiarchive.org/dandiset/000128 # replace URL with URL for dataset you want
# # URLS are:
# # - MC_Maze: https://dandiarchive.org/dandiset/000128
# # - MC_RTT: https://dandiarchive.org/dandiset/000129
# # - Area2_Bump: https://dandiarchive.org/dandiset/000127
# # - DMFC_RSG: https://dandiarchive.org/dandiset/000130
# # - MC_Maze_Large: https://dandiarchive.org/dandiset/000138
# # - MC_Maze_Medium: https://dandiarchive.org/dandiset/000139
# # - MC_Maze_Small: https://dandiarchive.org/dandiset/000140

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset-name",
        choices=["mc_maze", "mc_maze_large", "mc_maze_medium", "mc_maze_small", "mc_rtt", "area2_bump", "dmfc_rsg"],
        help="name of dataset to generate training and evaluation data",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    dataset_name = vars(args)['dataset-name']
    datapath_dict = {'mc_maze': './data/000128/sub-Jenkins/',
                     'mc_maze_large': './data/000138/sub-Jenkins/',
                     'mc_maze_medium': './data/000139/sub-Jenkins/',
                     'mc_maze_small': './data/000140/sub-Jenkins/',
                     'mc_rtt': './data/000129/sub-Indy/',
                     'area2_bump': './data/000127/sub-Han/',
                     'dmfc_rsg': './data/000130/sub-Haydn/',
                     }
    datapath = datapath_dict[f'{dataset_name}']
    ## Load data from NWB file:
    dataset = NWBDataset(datapath)

    ## Dataset preparation
    # Choose bin width and resample
    bin_width = 5
    dataset.resample(bin_width)

    # Create suffix for group naming later
    suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'

    # Choose the phase here, either 'val' for the Validation phase or 'test' for the Test phase
    # Note terminology overlap with 'train', 'val', and 'test' data splits -
    # the phase name corresponds to the data split that predictions are evaluated on
    for phase in ['val', 'test']:
        ## Make train data
        # Create input tensors, returned in dict form
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split=train_split, 
                                                include_behavior=True, include_forward_pred=True, save_file=True,
                                                save_path=f"./data/{dataset_name}_{'train' if phase=='val' else 'trainval'}.h5")

        ## Make eval data
        # Split for evaluation is same as phase name
        eval_split = phase
        # Make data tensors
        eval_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split=eval_split, save_file=True,
                                            save_path=f"./data/{dataset_name}_{phase}.h5")

        combine_h5([f"./data/{dataset_name}_{'train' if phase=='val' else 'trainval'}.h5", f"./data/{dataset_name}_{phase}.h5"], save_path=f"./data/{dataset_name}{'' if phase=='val' else '_test'}_full.h5")

    ## Make data to evaluate predictions with
    # Reset logging level to hide excessive info messages
    logging.getLogger().setLevel(logging.WARNING)

    phase = 'val'
    # If 'val' phase, make the target data
    if phase == 'val':
        # Note that the RTT task is not well suited to trial averaging, so PSTHs are not made for it
        target_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', 
                                                include_psth=True, save_file=True, save_path=f"./data/{dataset_name}_target.h5")


if __name__ == "__main__":
    main()

