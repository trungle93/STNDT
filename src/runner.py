#!/usr/bin/env python3
# Author: Joel Ye
# Original file available at https://github.com/snel-repo/neural-data-transformers/blob/master/src/runner.py
# Adapted by Trung Le
# Adapted training and evaluation pipeline with spatiotemporal attention and contrastive learning
# Added logging for co-bps metrics

import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional
from sklearn.metrics import r2_score, explained_variance_score
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_transformers import AdamW, WarmupCosineSchedule
from torch.utils import data
from scipy.special import gammaln
from torch.utils.tensorboard import SummaryWriter as TensorboardWriter

from third_party.src.logger_wrapper import create_logger
from third_party.src.utils import get_inverse_sqrt_schedule
from src.model import STNDT
from src.dataset import DATASET_MODES, SpikesDataset
from src.mask import Masker, UNMASKED_LABEL, DEFAULT_MASK_VAL

"""
Runner class for STNDT
"""

def get_lightest_gpus(num_gpus):
    # TODO update with better CUDA_VISIBLE_DEVICES support (or just use ray)
    if torch.cuda.device_count() == 1:
        return [0]
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)[-num_gpus:].tolist()

def exp_smooth(new_metric, old_metric, mu=0.5):
    r""" Higher mu is smoother """
    return (1.0 - mu) * new_metric + mu * old_metric

def exp_smooth_dict(new_metrics, rolling_metrics, mu=0.5):
    for m in new_metrics:
        if m in rolling_metrics:
            rolling_metrics[m] = exp_smooth(new_metrics[m], rolling_metrics[m], mu)

class Runner:
    r"""
        Two paths to inference.
        A:
            Have a config file.
            Load device.
            Load a checkpoint.
        B:
            Pass a checkpoint path (other steps automated)
        We prioritize path A.
    """
    def __init__(self, config=None, checkpoint_path=None, contrast_phase=True):
        assert config is not None or checkpoint_path is not None
        self.flush_secs = 10
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.num_neurons = 0
        self.pth_time = 0
        self.count_updates = 0
        self.count_checkpoints = 0
        self.num_gpus = 0
        self.masker = None
        self.contrast_masker = None
        self.rolling_metrics = {} # For PBT
        self.tuning_metric = {} # metric for selecting best model in tuning algorithm
        self.contrast_phase = contrast_phase

        if checkpoint_path is not None:
            if not torch.cuda.is_available():
                self.device = torch.device("cpu")
            ckpt_dict = torch.load(checkpoint_path, map_location=self.device)
            config = ckpt_dict["config"]
        self.config = config
        self.log_interval = self.config.TRAIN.LOG_INTERVAL
        self.val_interval = self.config.TRAIN.VAL_INTERVAL
        self.patience = self.config.TRAIN.PATIENCE
        if not osp.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR, exist_ok=True)
        logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
        self.logger = create_logger()
        self.logger.clear_filehandlers()
        self.logger.add_filehandler(logfile_path)
        if hasattr(config.TRAIN, "TUNE_MODE") and config.TRAIN.TUNE_MODE:
            self.logger.clear_streamhandlers()

        self.best_val = {
            "value": 1e9,
            "update": -1,
        }
        self.best_cobps = {
            "value": -10000,
            "update": -1,
        }
        self.best_contrast = {
            "value": 1e9,
            "update": -1,
        }
        self.best_unmasked_val = {
            "value": 1e9,
            "update": -1,
        }
        self.best_R2 = {
            "value": -100,
            "update": -1,
        }

        if checkpoint_path is not None:
            self.load_device()
            self.load_checkpoint(checkpoint_path, map_location=self.device)

    def setup_model(self, device):
        r""" Creates model and assigns to device """
        self.model = STNDT(
            self.config.MODEL,
            self.trial_length,
            self.num_neurons,
            device,
            max_spikes=self.max_spikes
        )
        num_hidden = self.model.get_hidden_size()
        if self.num_gpus > 1:
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_indices = get_lightest_gpus(self.num_gpus)
            else:
                gpu_indices = list(range(self.num_gpus))
            if self.device_gpu in gpu_indices:
                gpu_indices.remove(self.device_gpu)
            else:
                gpu_indices = gpu_indices[:-1]
            gpu_indices = [self.device_gpu] + gpu_indices # Make sure our primary gpu is first
            self.model = nn.DataParallel(self.model, device_ids=gpu_indices)
        self.model = self.model.to(device)
        return num_hidden

    def _get_parameters(self):
        return list(self.model.parameters())

    def _do_log(self, update):
        return (
            update > 0 and update % self.log_interval == 0
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optim_state": None if self.optimizer is None else self.optimizer.state_dict(),
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "config": self.config,
            "best_val": self.best_val,
            "best_cobps": self.best_cobps,
            "best_contrast": self.best_contrast,
            "best_unmasked_val": self.best_unmasked_val,
            "best_r2": self.best_R2,
            "max_spikes": self.max_spikes,
            "num_neurons": self.num_neurons,
            "trial_length": self.trial_length,
        }
        checkpoint["extra_state"] = dict( # metadata
            update=self.count_updates,
            checkpoint=self.count_checkpoints,
            pth_time=self.pth_time,
            max_spikes=self.max_spikes
        )

        if extra_state is not None:
            checkpoint["extra_state"].update(extra_state)

        if len(osp.split(file_name)[0]) > 0:
            full_path = file_name
        else:
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            full_path = osp.join(self.config.CHECKPOINT_DIR, file_name)
        torch.save(
            checkpoint, full_path
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Will fully load model if not already configured. Expects runner devices to be set.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        ckpt_dict = torch.load(checkpoint_path, *args, **kwargs)
        if "num_neurons" in ckpt_dict:
            self.num_neurons = ckpt_dict["num_neurons"]
        if "trial_length" in ckpt_dict:
            self.trial_length = ckpt_dict["trial_length"]
        if "max_spikes" in ckpt_dict:
            self.max_spikes = ckpt_dict["max_spikes"]
        if self.model is None:
            self.setup_model(self.device)
        self.model.load_state_dict(ckpt_dict["state_dict"])
        if "optim_state" in ckpt_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt_dict["optim_state"])
        if "lr_scheduler" in ckpt_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_dict["lr_scheduler"])
        if "best_val" in ckpt_dict:
            self.best_val = ckpt_dict["best_val"]
        if "best_cobps" in ckpt_dict:
            self.best_cobps = ckpt_dict["best_cobps"]
        if "best_contrast" in ckpt_dict:
            self.best_contrast = ckpt_dict["best_contrast"]
        if "best_unmasked_val" in ckpt_dict:
            self.best_unmasked_val = ckpt_dict["best_unmasked_val"]
        if "best_r2" in ckpt_dict:
            self.best_R2 = ckpt_dict["best_r2"]
        if "extra_state" in ckpt_dict:
            self.count_updates = ckpt_dict["extra_state"]["update"]
            self.logger.info("Update loaded -- {}".format(self.count_updates))
            self.count_checkpoints = ckpt_dict["extra_state"]["checkpoint"]
            self.pth_time = ckpt_dict["extra_state"]["pth_time"]
        return ckpt_dict

    def load_device(self):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.num_gpus = min(self.config.SYSTEM.NUM_GPUS, torch.cuda.device_count())
            self.logger.info(f"Using {self.num_gpus} GPUs")
            gpu_id = self.config.SYSTEM.TORCH_GPU_ID
            if self.config.SYSTEM.GPU_AUTO_ASSIGN:
                gpu_id = get_lightest_gpus(1)[0]
            self.device = (
                torch.device("cuda", gpu_id)
            )
            self.device_gpu = gpu_id

        self.logger.info(f"Using {self.device}")

    def update_config(self, config):
        r""" Update config node and propagate through model. Used for pbt.
        """
        if self.config.TRAIN.LR.INIT != config.TRAIN.LR.INIT and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = config.TRAIN.LR.INIT # Manualy override of LR
        self.config = config
        if self.masker is not None:
            self.masker.config = config.TRAIN
        self.model.update_config(config.MODEL)

    def load_train_val_data_and_masker(self):
        training_set = SpikesDataset(self.config, self.config.DATA.TRAIN_FILENAME, mode=DATASET_MODES.train, logger=self.logger)
        self.training_generator = data.DataLoader(training_set,
            batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=True
        )
        # We'll need this to embed spikes. Hoping max spikes for val/train isn't too far off
        self.max_spikes = training_set.get_max_spikes()
        self.logger.info(f"Clipping all spikes to {self.max_spikes}.")
        self.logger.info(f"Training on {len(training_set)} samples.")

        if self.config.TRAIN.DO_VAL:
            self.validation_set = SpikesDataset(self.config, self.config.DATA.VAL_FILENAME, mode=DATASET_MODES.val, logger=self.logger)
            self.validation_set.clip_spikes(self.max_spikes)
            self.validation_generator = data.DataLoader(self.validation_set,batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=False)

        self.num_neurons = training_set.get_num_neurons()
        self.trial_length = training_set.trial_length
        self.masker = Masker(self.config.TRAIN, self.device)
        self.contrast_masker = Masker(self.config.TRAIN, self.device)

    def load_optimizer(self, num_hidden):
        train_cfg = self.config.TRAIN
        # if is_learning_model(self.config.MODEL.NAME):
        self.optimizer = AdamW(
            list(filter(lambda p: p.requires_grad, self._get_parameters())),
            lr=train_cfg.LR.INIT,
            weight_decay=train_cfg.WEIGHT_DECAY,
            eps=train_cfg.EPS,
        )

        self.logger.info(
            "number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.model.parameters()
                    if param.requires_grad
                )
            )
        )

        if self.optimizer is not None and train_cfg.LR.SCHEDULE:
            if train_cfg.LR.SCHEDULER == "cosine":
                self.lr_scheduler = WarmupCosineSchedule(
                    self.optimizer,
                    warmup_steps=train_cfg.LR.WARMUP,
                    t_total=train_cfg.NUM_UPDATES
                )
            else:
                self.lr_scheduler = get_inverse_sqrt_schedule(
                    self.optimizer,
                    warmup_steps=train_cfg.LR.WARMUP,
                    lr_max=train_cfg.LR.INIT
                )

    def train(self, checkpoint_path=None) -> None:
        r"""Main method for training model.

        Args:
            checkpoint_path: path of checkpoint to load
        Returns:
            None
        """
        self.load_device()
        train_cfg = self.config.TRAIN

        self.load_train_val_data_and_masker()
        num_hidden = self.setup_model(self.device)
        self.load_optimizer(num_hidden)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, map_location=self.device)

        start_updates = self.count_updates

        for update in range(start_updates, train_cfg.NUM_UPDATES):
            metrics = self.train_epoch()
            if metrics["done"]:
                break
            torch.cuda.empty_cache()
        if not metrics["done"]:
           self.logger.info("Reached max updates without early stopping. Consider training some more.")

        if not train_cfg.TUNE_MODE:
            metrics_dict = {
                "Loss": self.best_val["value"],
                "Unmasked Loss": self.best_unmasked_val["value"],
                "Cobps": self.best_cobps["value"],
                "Contrast": self.best_contrast["value"],
            }
            if train_cfg.DO_R2:
                metrics_dict.update({ "R2": self.best_R2["value"] })
            with TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                writer.add_hparams(self.extract_hps_dict(), metrics_dict)
        torch.cuda.empty_cache()

    def train_epoch(self):
        r"""
            One (PBT) epoch of training. Model and data should be set up and on device at this point.

            Note: LFADS runs an epoch every pass through the data. This may be too frequently for transformers.
            i.e. we may need to do multiple passes through the data. For now, we're changing to report every pass through data.

            Returns:
                metrics: Information about the epoch.
                    "done" -- should stop this run (e.g. due to early stopping). Keyword for Tune PBT.
        """
        if self.training_generator is None:
            raise Exception("No dataset generator set")

        update = self.count_updates
        train_cfg = self.config.TRAIN

        expand_prob = min((update - train_cfg.MASK_SPAN_RAMP_START) / (train_cfg.MASK_SPAN_RAMP_END - train_cfg.MASK_SPAN_RAMP_START), 1)
        self.model.train()

        t_start = time.time()
        for spikes, rates, heldout_spikes, forward_spikes in self.training_generator:
            spikes = spikes.to(self.device)
            rates = rates.to(self.device) if self.config.MODEL.REQUIRES_RATES else None
            if self.training_generator.dataset.has_heldout:
                heldout_spikes = heldout_spikes.to(self.device)
                forward_spikes = forward_spikes.to(self.device)
            else:
                heldout_spikes = None
                forward_spikes = None
            masked_spikes, labels, masked_heldin, masked_batch_full, labels_full, spikes_full = self.masker.mask_batch(
                spikes,
                contrast_mode=False,
                max_spikes=self.max_spikes,
                should_mask=True,
                expand_prob=expand_prob,
                heldout_spikes=heldout_spikes,
                forward_spikes=forward_spikes
            )
            if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                contrast_masked_spikes1, _, _, _, _, _ = self.contrast_masker.mask_batch(
                    spikes,
                    contrast_mode=True,
                    max_spikes=self.max_spikes,
                    should_mask=True,
                    expand_prob=expand_prob,
                    heldout_spikes=heldout_spikes,
                    forward_spikes=forward_spikes
                )
                contrast_masked_spikes2, _, _, _, _, _ = self.contrast_masker.mask_batch(
                    spikes,
                    contrast_mode=True,
                    max_spikes=self.max_spikes,
                    should_mask=True,
                    expand_prob=expand_prob,
                    heldout_spikes=heldout_spikes,
                    forward_spikes=forward_spikes
                )
            else:
                contrast_masked_spikes1 = None
                contrast_masked_spikes2 = None

            mlm_loss, masked_decoder_loss, contrast_loss, *_ = self.model(
                masked_spikes,
                labels,
                contrast_src1=contrast_masked_spikes1,
                contrast_src2=contrast_masked_spikes2,
                val_phase=False,
                rates=rates,
                return_outputs=True,
                return_weights=False,
            )
            if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                loss = mlm_loss.mean()
            else:
                loss = masked_decoder_loss.mean()

            if self.optimizer is not None:

                self.optimizer.zero_grad()
                loss.backward()
                params = self._get_parameters()

                nn.utils.clip_grad_norm_(
                    params, train_cfg.MAX_GRAD_NORM
                )
                self.optimizer.step()

        self.pth_time += time.time() - t_start
        self.count_updates += 1
        update = self.count_updates

        if self.optimizer is not None and train_cfg.LR.SCHEDULE:
            self.lr_scheduler.step()

        if self._do_log(update):
            # * Note we're only logging the loss of the last train step
            with TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                if self.optimizer is not None and train_cfg.LR.SCHEDULE:
                    writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0])
                    self.logger.queue_stat("LR", self.lr_scheduler.get_last_lr()[0])
                writer.add_scalar(
                    "loss", # train loss
                    loss,
                    update,
                )

            self.logger.queue_stat("loss", loss.item())

        metrics_dict = dict(
            done = False,
            epoch = self.count_updates,
            best_masked_loss = self.best_val["value"], # Tune will reference this value to select best model.
            best_cobps = self.best_cobps["value"], # Tune will reference this value to select best model.
            best_contrast = self.best_contrast["value"], # Tune will reference this value to select best model.
        )

        torch.cuda.empty_cache()
        if (train_cfg.DO_VAL and update % self.val_interval == 0):
            self.model.eval()
            with torch.no_grad():
                contrast_losses = []
                val_losses = []
                no_mask_losses = []
                pred_rates = []
                heldout_spikes_full = []
                for spikes, rates, heldout_spikes, forward_spikes in self.validation_generator:
                    spikes = spikes.to(self.device)
                    rates = rates.to(self.device)
                    if self.validation_set.has_heldout:
                        heldout_spikes = heldout_spikes.to(self.device)
                        forward_spikes = forward_spikes.to(self.device)
                    else:
                        heldout_spikes = None
                        forward_spikes = None
                    feed_rates = rates if self.config.MODEL.REQUIRES_RATES else None
                    masked_spikes, labels, _, _, _, _ = self.masker.mask_batch(
                        spikes,
                        contrast_mode=False,
                        max_spikes=self.max_spikes,
                        should_mask=True,
                        heldout_spikes=heldout_spikes,
                        forward_spikes=forward_spikes,
                    )
                    if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                        contrast_masked_spikes1, _, _, _, _, _ = self.contrast_masker.mask_batch(
                            spikes,
                            contrast_mode=True,
                            max_spikes=self.max_spikes,
                            should_mask=True,
                            expand_prob=expand_prob,
                            heldout_spikes=heldout_spikes,
                            forward_spikes=forward_spikes,
                        )
                        contrast_masked_spikes2, _, _, _, _, _ = self.contrast_masker.mask_batch(
                            spikes,
                            contrast_mode=True,
                            max_spikes=self.max_spikes,
                            should_mask=True,
                            expand_prob=expand_prob,
                            heldout_spikes=heldout_spikes,
                            forward_spikes=forward_spikes,
                        )
                    else:
                        contrast_masked_spikes1 = None
                        contrast_masked_spikes2 = None

                    loss, masked_decoder_loss, contrast_loss, *_ = self.model(
                        masked_spikes,
                        labels,
                        contrast_src1=contrast_masked_spikes1,
                        contrast_src2=contrast_masked_spikes2,
                        val_phase=True,
                        rates=feed_rates,
                        return_outputs=True,
                        return_weights=False,
                    )

                    val_losses.append(masked_decoder_loss)
                    contrast_losses.append(contrast_loss)

                    # no_mask evaluation should still exclude heldout neurons
                    if heldout_spikes is not None:
                        no_mask_labels = spikes.clone()
                        no_mask_labels = torch.cat([no_mask_labels, torch.zeros_like(heldout_spikes)], -1)
                        no_mask_labels = torch.cat([no_mask_labels, torch.zeros_like(forward_spikes)], 1)
                        no_mask_labels[:, :, -heldout_spikes.size(-1):] = -100 # unmasked_label
                        no_mask_labels[:, -forward_spikes.size(1):,:] = -100 # unmasked_label
                        spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
                        spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)], 1)
                    else:
                        no_mask_labels = spikes
                    no_mask_loss, masked_decoder_loss, _, batch_rates, *_ = self.model(
                        spikes,
                        no_mask_labels,
                        contrast_src1=None,
                        contrast_src2=None,
                        val_phase=True,
                        passthrough=True,
                        rates=rates,
                    )
                    no_mask_losses.append(masked_decoder_loss)
                    pred_rates.append(batch_rates)
                    heldout_spikes_full.append(heldout_spikes)

                contrast_loss = torch.cat(contrast_losses, dim=0).mean()
                val_loss = torch.cat(val_losses, dim=0).mean()
                no_mask_loss = torch.cat(no_mask_losses, dim=0).mean()
                heldout_spikes = torch.cat(heldout_spikes_full, dim=0)
                pred_rates = torch.cat(pred_rates, dim=0)

                if self.config.TRAIN.DO_CONTRAST and self.contrast_phase and contrast_loss.item() < self.best_contrast["value"]:
                    self.logger.info(f"Overwriting best contrast loss {self.best_contrast['value']} from {self.best_contrast['update']} with {contrast_loss.item()} at {update}.")
                    self.best_contrast["value"] = contrast_loss.item()
                    self.best_contrast["update"] = update
                    self.save_checkpoint(f'{self.config.VARIANT}.contrast.pth')

                eval_rates_heldout = torch.exp(pred_rates.clone()[:, :heldout_spikes.size(1), -heldout_spikes.size(-1):])
                eval_spikes_heldout = heldout_spikes.clone().type(torch.FloatTensor).to(self.device)

                cobps = float(bits_per_spike(eval_rates_heldout.to('cpu').numpy()[()].astype('float'), eval_spikes_heldout.to('cpu').numpy()[()].astype('float')))
                metrics_dict["cobps"] = cobps
                if cobps > self.best_cobps["value"]:
                    self.logger.info(f"Overwriting best co-bps {self.best_cobps['value']} from {self.best_cobps['update']} with {cobps} at {update}.")
                    self.best_cobps["value"] = cobps
                    self.best_cobps["update"] = update
                    if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                        self.save_checkpoint(f'{self.config.VARIANT}.contrast.cobps.pth')
                    else:
                        self.save_checkpoint(f'{self.config.VARIANT}.cobps.pth')

                metrics_dict["best_cobps"] = self.best_cobps["value"]

                no_mask_loss = no_mask_loss.mean()

                metrics_dict["unmasked_loss"] = no_mask_loss.item()
                metrics_dict["masked_loss"] = val_loss.item()

                if "smth_masked_loss" not in self.rolling_metrics:
                    self.rolling_metrics["smth_masked_loss"] = metrics_dict["masked_loss"]
                else:
                    self.rolling_metrics["smth_masked_loss"] = exp_smooth(metrics_dict["masked_loss"], self.rolling_metrics["smth_masked_loss"])
                metrics_dict["smth_masked_loss"] = self.rolling_metrics["smth_masked_loss"]

                if self._do_log(update):
                    with TensorboardWriter(
                        self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
                    ) as writer:
                        writer.add_scalar(
                            "val_loss",
                            val_loss,
                            update,
                        )
                        writer.add_scalar(
                            "cobps",
                            cobps,
                            update,
                        )

                        writer.add_scalar(
                            "unmasked_loss",
                            no_mask_loss,
                            update,
                        )
                        self.logger.queue_stat("val loss", val_loss.item())
                        self.logger.queue_stat("cobps", cobps)
                        self.logger.queue_stat("unmasked val loss", no_mask_loss.item())
                        if train_cfg.DO_R2 and self.validation_set.has_rates:
                            r2 = self.neuron_r2(rates, pred_rates)
                            writer.add_scalar("r2", r2, update)
                            self.logger.queue_stat("r2", r2)
                            if self.best_R2["value"] < r2:
                                self.best_R2["value"] = r2
                                self.best_R2["update"] = update
                                self.save_checkpoint(f'{self.config.VARIANT}.gr2.pth') # greatest r2
                            metrics_dict["r2"] = r2

                if no_mask_loss.item() < self.best_unmasked_val["value"]:
                    self.logger.info(f"Overwriting best unmasked val {self.best_unmasked_val['value']} from {self.best_unmasked_val['update']} with {no_mask_loss} at {update}.")
                    self.best_unmasked_val["value"] = no_mask_loss.item()
                    self.best_unmasked_val["update"] = update
                    if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                        self.save_checkpoint(f'{self.config.VARIANT}.contrast.lfve.pth')
                    else:
                        self.save_checkpoint(f'{self.config.VARIANT}.lfve.pth')
                if val_loss.item() < self.best_val["value"]:
                    self.logger.info(f"Overwriting best val {self.best_val['value']} from {self.best_val['update']} with {val_loss} at {update}.")
                    self.best_val["value"] = val_loss.item()
                    self.best_val["update"] = update
                    if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                        self.save_checkpoint(f'{self.config.VARIANT}.contrast.lve.pth')
                    else:
                        self.save_checkpoint(f'{self.config.VARIANT}.lve.pth')

                if update - self.best_val["update"] > self.patience and update - self.best_cobps["update"] > self.patience:                    
                    self.logger.info(f"Val loss or unmasked val loss or cobps has not improved for {self.patience} updates. Stopping...")
                    self.logger.info(f"Best val: {self.best_val['value']} at {self.best_val['update']} updates.")
                    self.logger.info(f"Best unmasked val: {self.best_unmasked_val['value']} at {self.best_unmasked_val['update']} updates.")
                    self.logger.info(f"Best cobps: {self.best_cobps['value']} at {self.best_cobps['update']} updates.")
                    if train_cfg.DO_R2 and self.validation_set.has_rates: # log down for hparams
                        self.logger.info(f"Best R2: {self.best_R2['value']} at {self.best_R2['update']}")
                        r2 = self.neuron_r2(rates, pred_rates)
                        metrics_dict["r2"] = r2
                    metrics_dict["done"] = True

                metrics_dict["best_masked_loss"] = self.best_val["value"]

        if self._do_log(update):
            self.logger.log_update(update)
            self.logger.info(
                "update: {}\tpth-time: {:.3f}s\t".format(
                    update, self.pth_time
                )
            )

        if update % train_cfg.CHECKPOINT_INTERVAL == 0 and not train_cfg.TUNE_MODE: # Don't save extra checkpoints when sweeping

            if self.config.TRAIN.DO_CONTRAST and self.contrast_phase:
                self.save_checkpoint(f"{self.config.VARIANT}.contrast.{self.count_checkpoints}.pth")
            else:
                self.save_checkpoint(f'{self.config.VARIANT}.{self.count_checkpoints}.pth')
            self.count_checkpoints += 1

        torch.cuda.empty_cache()
        return metrics_dict


    def get_rates(
        self,
        checkpoint_path = None,
        mode = DATASET_MODES.trainval,
        save_path = None,
        keep_layers = -1, # keep last layer
    ) -> None:
        r"""Evaluates model (with checkpoint loaded) on train/val data and retrieves rates and activations (features for downstream tasks).
        Matches LFADS structure - we thus use a single dataset (no train val differentiator).
        Args:
            checkpoint_path: path of checkpoint (will use model on runner if not provided)
            save_path: Path to save activations at (optional). Does not save if nothing provided
            mode: train/val/test
        Returns:
            pred_rates: rates prediction
            all_s_attentions: spatial attention weights of all transformer layers
            all_s_attentions: temporal attention weights of all transformer layers
            layer_outputs: outputs of all transformer layers
        """
        self.logger.info(f"Getting rates...")
        if self.device is None:
            self.load_device()
        train_cfg = self.config.TRAIN
        self.masker = Masker(train_cfg, self.device) # Unused

        whole_set = SpikesDataset(self.config, self.config.DATA.TRAIN_FILENAME, mode=mode, logger=self.logger)
        self.max_spikes = whole_set.get_max_spikes()
        self.num_neurons = whole_set.get_num_neurons()
        self.logger.info(f"Evaluating on {len(whole_set)} samples.")
        data_generator = data.DataLoader(whole_set,
            batch_size=train_cfg.BATCH_SIZE, shuffle=False
        )

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        if self.num_neurons is None:
            self.num_neurons(whole_set.get_num_neurons())
        update = ckpt_dict["extra_state"]["update"]
        if self.max_spikes is not None:
            whole_set.clip_spikes(self.max_spikes)
        self.model.eval()

        with torch.no_grad():
            losses = []
            pred_rates = []
            layer_outputs = []
            all_s_attentions = []
            all_t_attentions = []
            for spikes, _, heldout_spikes, forward_spikes in data_generator:
                spikes = spikes.to(self.device)
                if data_generator.dataset.has_heldout:
                    heldout_spikes = heldout_spikes.to(self.device)
                    forward_spikes = forward_spikes.to(self.device)
                    # Do NOT provide privileged eval info
                    spikes_full = torch.cat([spikes.clone(), heldout_spikes], -1)
                    spikes_full = torch.cat([spikes_full, forward_spikes], 1)
                    spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
                    spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)], 1)
                else:
                    heldout_spikes = None
                    forward_spikes = None
                labels = spikes # i.e. predict everything
                loss, _, _, batch_rates, batch_attn_list, batch_layer_outputs = self.model(
                    spikes,
                    spikes,
                    contrast_src1=None,
                    contrast_src2=None,
                    val_phase=True,
                    passthrough=True,
                    return_outputs=True,
                    return_weights=True,
                )
                losses.append(loss.mean().item())
                pred_rates.append(batch_rates)
                batch_s_attn_list = [layer_tuple[0] for layer_tuple in batch_attn_list]
                batch_t_attn_list = [layer_tuple[1] for layer_tuple in batch_attn_list]
                all_s_attentions.append(batch_s_attn_list)
                all_t_attentions.append(batch_t_attn_list)
                layer_outputs.append(batch_layer_outputs)
            pred_rates = torch.cat(pred_rates, dim=0)
            if self.config.MODEL.LOGRATE:
                pred_rates = pred_rates.exp()
            s_attention_per_layer = zip(*all_s_attentions) # Lists of all samples, grouped by layer
            all_s_attentions = torch.stack([torch.cat(layer, dim=0) for layer in s_attention_per_layer], dim=0)
            t_attention_per_layer = zip(*all_t_attentions) # Lists of all samples, grouped by layer
            all_t_attentions = torch.stack([torch.cat(layer, dim=0) for layer in t_attention_per_layer], dim=0)
            layer_outputs = torch.cat(layer_outputs, dim=0)
            self.logger.queue_stat("test loss", torch.tensor(losses).mean().item())
            self.logger.log_update(update)

        if save_path is not None:
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('rates', data=pred_rates.cpu().numpy())
                f.create_dataset('layer_outputs', data=all_layer_outputs[-1].cpu().numpy()) # Only final layer
        return pred_rates, all_s_attentions, all_t_attentions, layer_outputs


    def _clean_rates(self, gt, pred, flatten=False):
        if gt.size() != pred.size():
            raise Exception(f"Incompatible r2 sizes, GT: {gt.size()}, Pred: {pred.size()}")

        if flatten or len(gt.size()) > 1:
            gt = gt.flatten(end_dim=1)
            pred = pred.flatten(end_dim=1)

        if self.config.MODEL.LOGRATE:
            gt = gt.exp()
            pred = pred.exp()
        return gt.cpu(), pred.cpu()

    def neuron_r2(self, gt, pred, **kwargs):
        gt, pred = self._clean_rates(gt, pred, **kwargs)
        return r2_score(gt, pred, multioutput='uniform_average')

    def neuron_vaf(self, gt, pred, **kwargs):
        gt, pred = self._clean_rates(gt, pred, **kwargs)
        return explained_variance_score(gt, pred, multioutput='uniform_average')

    # For HParams
    def extract_hps_dict(self):
        hp_dict = {}
        hp_dict.update(self._extract_flat_dict(self.config.MODEL, "MODEL"))
        hp_dict.update(self._extract_flat_dict(self.config.TRAIN, "TRAIN"))
        return hp_dict

    BLACKLIST = ['MODEL/LOSS']
    def _extract_flat_dict(self, config, prefix):
        flat_dict = {}
        if prefix in Runner.BLACKLIST:
            return flat_dict
        for key, value in config.items():
            if isinstance(value, dict):
                flat_dict.update(self._extract_flat_dict(value, f"{prefix}/{key}"))
            elif not isinstance(value, list): # drop lists
                flat_dict[f"{prefix}/{key}"] = value
        return flat_dict


def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)
    
    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate 
        predictions or not
    
    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert spikes.shape == rates.shape, \
        f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]
    
    assert not np.any(np.isnan(rates)), \
        "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), \
        "neg_log_likelihood: Negative rate predictions found"
    if (np.any(rates == 0)):
        rates[rates == 0] = 1e-9
    
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts
    
    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    nll_null = neg_log_likelihood(np.tile(np.nanmean(spikes, axis=(0,1), keepdims=True), (spikes.shape[0], spikes.shape[1], 1)), spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

