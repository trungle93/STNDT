# Author: Joel Ye
# Original file available at https://github.com/snel-repo/neural-data-transformers/blob/master/scripts/analyze_utils.py
# Adapted by Trung Le based on AEStudio's extension: https://github.com/agencyenterprise/ae-nlb-2021/blob/39d0de79aef2b997dcb419a9d3f9cd81180ee57b/src/inference.py#L10
# Added a hook in init_by_ckpt() to change data file name from runner

import os
import os.path as osp
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

from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES


def setup_dataset(runner):
    r"""
    Called by init_by_ckpt() to set up dataset
    Args:
        runner: runner object
    Returns:
        spikes: heldin spikes
        rates: heldin rates (not applicable for NLB data)
        heldout_spikes
        forward_spikes
    """
    test_set = SpikesDataset(runner.config, runner.config.DATA.TRAIN_FILENAME, logger=runner.logger)
    runner.logger.info(f"Evaluating on {len(test_set)} samples.")
    test_set.clip_spikes(runner.max_spikes)
    spikes, rates, heldout_spikes, forward_spikes = test_set.get_dataset()
    if heldout_spikes is not None:
        heldout_spikes = heldout_spikes.to(runner.device)
    if forward_spikes is not None:
        forward_spikes = forward_spikes.to(runner.device)
    return spikes.to(runner.device), rates.to(runner.device), heldout_spikes, forward_spikes

def init_by_ckpt(ckpt_path, mode=DATASET_MODES.val, data_file=None):
    r"""
    Initializes runner from model checkpoint. To be used in inference and ensembling phases.
    Args:
        ckpt_path: path to saved checkpoint
        mode: mode (train/val/test) to load data split
        data_file: path to data file
    Returns:
        runner: runner object
        spikes: heldin spikes
        rates: heldin rates (not applicable for NLB data)
        heldout_spikes
        forward_spikes
    """
    runner = Runner(checkpoint_path=ckpt_path)
    if data_file is not None:
        runner.config.merge_from_list(['DATA.TRAIN_FILENAME', str(data_file)])
    runner.model.eval()
    torch.set_grad_enabled(False)
    spikes, rates, heldout_spikes, forward_spikes = setup_dataset(runner)
    return runner, spikes, rates, heldout_spikes, forward_spikes
