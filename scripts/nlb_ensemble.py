#%%

# 1. Load model and get rate predictions
import os
import os.path as osp
from pathlib import Path
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils import data

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import save_to_h5

from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from src.mask import UNMASKED_LABEL

from scripts.analyze_utils import init_by_ckpt
import argparse

DATA_DIR = Path("./data/")
ENSEMBLE_RESULTS_DIR = Path("./ensemble_results/")
RAY_RESULTS_DIR = Path("./ray_results/")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset-name",
        choices=["mc_maze", "mc_maze_large", "mc_maze_medium", "mc_maze_small", "mc_rtt", "area2_bump", "dmfc_rsg"],
        help="name of dataset to perform ensembling",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    variant = vars(args)['dataset-name']
    # variant = "mc_maze"
    # variant = "area2_bump"
    # variant = "dmfc_rsg"
    # variant = "mc_rtt"

    target_path = osp.join(DATA_DIR, f"{variant}_target.h5")

    with h5py.File(target_path, 'r') as h5file:
        target_dict = {f'{variant}': {key: h5file[f'{variant}'][key][()].astype(np.bool_) if key == 'train_decode_mask' or key == 'eval_decode_mask'
                        else h5file[f'{variant}'][key][()].astype(np.object) if key == 'eval_cond_idx'
                        else h5file[f'{variant}'][key][()].astype(np.int_) if key == 'eval_jitter'
                        else h5file[f'{variant}'][key][()].astype(np.float32)
                        for key in h5file[f'{variant}'].keys()
                        }}

    ray_results_dir = osp.join(RAY_RESULTS_DIR, f"{variant}_lite/{variant}_lite/")

    date = 'jan1'
    ckpt_path_list = []
    cobps_list = []
    velr2_list = []
    psthr2_list = []
    fpbps_list = []
    train_rates_heldin_list = []
    train_rates_heldout_list = []
    eval_rates_heldin_list = []
    eval_rates_heldout_list = []
    eval_rates_heldin_forward_list = []
    eval_rates_heldout_forward_list = []

    for root, dirs, files in os.walk(ray_results_dir):
        for file in files:
            if file.endswith("pth"):
                ckpt_path = os.path.join(root, file)
                print('evaluating', ckpt_path)
                runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)

                eval_rates, *_ = runner.get_rates(
                    checkpoint_path=ckpt_path,
                    save_path = None,
                    mode = DATASET_MODES.val
                )
                train_rates, *_ = runner.get_rates(
                    checkpoint_path=ckpt_path,
                    save_path = None,
                    mode = DATASET_MODES.train
                )

                eval_rates.cpu()
                train_rates.cpu()

                eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
                eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
                train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
                eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
                train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

                output_dict = {
                    variant: {
                        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
                        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
                        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
                        'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
                        'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
                        'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
                    }
                }
                eval_dict = evaluate(target_dict, output_dict)
                print(eval_dict)
                ckpt_path_list.append(ckpt_path)
                cobps_list.append(eval_dict[0][f'{variant}_split']['co-bps'])
                if variant!='dmfc_rsg':
                    velr2_list.append(eval_dict[0][f'{variant}_split']['vel R2'])
                else:
                    velr2_list.append(eval_dict[0][f'{variant}_split']['tp corr'])
                if variant!='mc_rtt':
                    psthr2_list.append(eval_dict[0][f'{variant}_split']['psth R2'])
                fpbps_list.append(eval_dict[0][f'{variant}_split']['fp-bps'])

                train_rates_heldin_list.append(train_rates_heldin)
                train_rates_heldout_list.append(train_rates_heldout)
                eval_rates_heldin_list.append(eval_rates_heldin)
                eval_rates_heldout_list.append(eval_rates_heldout)
                eval_rates_heldin_forward_list.append(eval_rates_heldin_forward)
                eval_rates_heldout_forward_list.append(eval_rates_heldout_forward)


    max_ensemble_size = 50
    best_ensemble_metrics = -1e3
    sort_idx_full = np.argsort(np.array(cobps_list))[::-1].tolist()
    ckpt_path_list_sorted_full = [ckpt_path_list[i] for i in sort_idx_full]
    print('checkpoints sorted by co-bps: ', ckpt_path_list_sorted_full)

    for ensemble_size in range(1, max_ensemble_size+1):
        print('evaluating ensemble size =', ensemble_size)     
        sort_idx = sort_idx_full[:ensemble_size]
        ckpt_path_list_sorted = ckpt_path_list_sorted_full[:ensemble_size]
        train_rates_heldin_ensemble = torch.stack([train_rates_heldin_list[i] for i in sort_idx], dim=0).mean(dim=0)
        train_rates_heldout_ensemble = torch.stack([train_rates_heldout_list[i] for i in sort_idx], dim=0).mean(dim=0)
        eval_rates_heldin_ensemble = torch.stack([eval_rates_heldin_list[i] for i in sort_idx], dim=0).mean(dim=0)
        eval_rates_heldout_ensemble = torch.stack([eval_rates_heldout_list[i] for i in sort_idx], dim=0).mean(dim=0)
        eval_rates_heldin_forward_ensemble = torch.stack([eval_rates_heldin_forward_list[i] for i in sort_idx], dim=0).mean(dim=0)
        eval_rates_heldout_forward_ensemble = torch.stack([eval_rates_heldout_forward_list[i] for i in sort_idx], dim=0).mean(dim=0)
        eval_rates_ensemble = torch.cat([eval_rates_heldin_ensemble, eval_rates_heldout_ensemble], dim=2)
        eval_rates_forward_ensemble = torch.cat([eval_rates_heldin_forward_ensemble, eval_rates_heldout_forward_ensemble], dim=2)
        eval_rates_full_ensemble = torch.cat([eval_rates_ensemble, eval_rates_forward_ensemble], dim=1)

        output_dict = {
            variant: {
                'train_rates_heldin': train_rates_heldin_ensemble.cpu().numpy(),
                'train_rates_heldout': train_rates_heldout_ensemble.cpu().numpy(),
                'eval_rates_heldin': eval_rates_heldin_ensemble.cpu().numpy(),
                'eval_rates_heldout': eval_rates_heldout_ensemble.cpu().numpy(),
                'eval_rates_heldin_forward': eval_rates_heldin_forward_ensemble.cpu().numpy(),
                'eval_rates_heldout_forward': eval_rates_heldout_forward_ensemble.cpu().numpy()
            }
        }
        eval_dict = evaluate(target_dict, output_dict)
        print(eval_dict)
        if eval_dict[0][f'{variant}_split']['co-bps'] > best_ensemble_metrics:
            best_ensemble_metrics = eval_dict[0][f'{variant}_split']['co-bps']
            best_ensemble_size = ensemble_size
    print('best ensemble co-bps =', best_ensemble_metrics, 'at ensemble size =', best_ensemble_size)


    ##### test:
    print('getting ensembled prediction for test split ...')
    test_data_file = osp.join(DATA_DIR, f"{variant}_test_full.h5")
    train_rates_heldin_list = []
    train_rates_heldout_list = []
    eval_rates_heldin_list = []
    eval_rates_heldout_list = []
    eval_rates_heldin_forward_list = []
    eval_rates_heldout_forward_list = []
    for ensemble_size in [1, best_ensemble_size]:
        print('evaluating ensemble size = ', ensemble_size)
        sort_idx = sort_idx_full[:ensemble_size]
        ckpt_path_list_sorted = ckpt_path_list_sorted_full[:ensemble_size]
        print('checkpoints sorted by co-bps:', ckpt_path_list_sorted)

        for ckpt_path in ckpt_path_list_sorted:
            runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val, data_file=test_data_file) # mode actually not needed

            eval_rates, *_ = runner.get_rates(
                checkpoint_path=ckpt_path,
                save_path = None,
                mode = DATASET_MODES.val
            )
            train_rates, *_ = runner.get_rates(
                checkpoint_path=ckpt_path,
                save_path = None,
                mode = DATASET_MODES.train
            )

            eval_rates.cpu()
            train_rates.cpu()

            eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
            eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
            train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
            eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
            train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

            train_rates_heldin_list.append(train_rates_heldin)
            train_rates_heldout_list.append(train_rates_heldout)
            eval_rates_heldin_list.append(eval_rates_heldin)
            eval_rates_heldout_list.append(eval_rates_heldout)
            eval_rates_heldin_forward_list.append(eval_rates_heldin_forward)
            eval_rates_heldout_forward_list.append(eval_rates_heldout_forward)


        train_rates_heldin_ensemble = torch.stack(train_rates_heldin_list, dim=0).mean(dim=0)
        train_rates_heldout_ensemble = torch.stack(train_rates_heldout_list, dim=0).mean(dim=0)
        eval_rates_heldin_ensemble = torch.stack(eval_rates_heldin_list, dim=0).mean(dim=0)
        eval_rates_heldout_ensemble = torch.stack(eval_rates_heldout_list, dim=0).mean(dim=0)
        eval_rates_heldin_forward_ensemble = torch.stack(eval_rates_heldin_forward_list, dim=0).mean(dim=0)
        eval_rates_heldout_forward_ensemble = torch.stack(eval_rates_heldout_forward_list, dim=0).mean(dim=0)
        eval_rates_ensemble = torch.cat([eval_rates_heldin_ensemble, eval_rates_heldout_ensemble], dim=2)
        eval_rates_forward_ensemble = torch.cat([eval_rates_heldin_forward_ensemble, eval_rates_heldout_forward_ensemble], dim=2)
        eval_rates_full_ensemble = torch.cat([eval_rates_ensemble, eval_rates_forward_ensemble], dim=1)
        

        output_dict = {
            variant: {
                'train_rates_heldin': train_rates_heldin_ensemble.cpu().numpy(),
                'train_rates_heldout': train_rates_heldout_ensemble.cpu().numpy(),
                'eval_rates_heldin': eval_rates_heldin_ensemble.cpu().numpy(),
                'eval_rates_heldout': eval_rates_heldout_ensemble.cpu().numpy(),
                'eval_rates_heldin_forward': eval_rates_heldin_forward_ensemble.cpu().numpy(),
                'eval_rates_heldout_forward': eval_rates_heldout_forward_ensemble.cpu().numpy()
            }
        }

        save_to_h5(output_dict, osp.join(ENSEMBLE_RESULTS_DIR, f'{variant}_{ensemble_size}_{date}.h5'))


if __name__ == "__main__":
    main()
