#!/usr/bin/env python3
# Author: Joel Ye
# Original file available at https://github.com/snel-repo/neural-data-transformers/blob/master/src/mask.py
# Adapted by Trung Le
# Created separate masks for contrastive mode

import torch
import torch.nn as nn
import torch.nn.functional as F

# Some infeasibly high spike count
DEFAULT_MASK_VAL = 30
UNMASKED_LABEL = -100
SUPPORTED_MODES = ["full", "timestep", "neuron", "timestep_only"]

# Use a class so we can cache random mask
class Masker:
    def __init__(self, train_cfg, device):
        self.update_config(train_cfg)
        if self.cfg.MASK_MODE not in SUPPORTED_MODES:
            raise Exception(f"Given {self.cfg.MASK_MODE} not in supported {SUPPORTED_MODES}")
        if self.cfg.CONTRAST_MASK_MODE not in SUPPORTED_MODES:
            raise Exception(f"Given {self.cfg.CONTRAST_MASK_MODE} not in supported {SUPPORTED_MODES}")
        self.device = device

    def update_config(self, config):
        self.cfg = config
        self.prob_mask = None

    def expand_mask(self, mask, width):
        r"""
            args:
                mask: N x T
                width: expansion block size
        """
        kernel = torch.ones(width, device=mask.device).view(1, 1, -1)
        expanded_mask = F.conv1d(mask.unsqueeze(1), kernel, padding= width// 2).clamp_(0, 1)
        if width % 2 == 0:
            expanded_mask = expanded_mask[...,:-1] # crop if even (we've added too much padding)
        return expanded_mask.squeeze(1)

    def mask_batch(
        self,
        batch,
        contrast_mode,
        mask=None,
        max_spikes=DEFAULT_MASK_VAL - 1,
        should_mask=True,
        expand_prob=0.0,
        heldout_spikes=None,
        forward_spikes=None,
    ):
        r""" Given complete batch, mask random elements and return true labels separately.
        Modifies batch OUT OF place!
        Modeled after HuggingFace's `mask_tokens` in `run_language_modeling.py`
        args:
            batch: heldin spikes with shape NxTxH
            contrast_mode: if True, use configs for mask in contrastive phase
            mask: optional custom mask to use
            max_spikes: max number of spikes at a given timestep. In case not zero masking, "mask token"
            expand_prob: expand mask with this probability
            heldout_spikes: heldout portion
            forward_spikes: forward portion
        returns:
            batch: masked heldin portion concatenated with all-zero heldout and forward portions
            labels: unmasked heldin portion concatenated with true heldout and forward portions
            masked_heldin: masked heldin portion
            batch_full: consists of heldin, heldout, forward spikes all corrupted by mask
            labels_full: unmasked heldin, heldout, forward portions
            spikes_full: true heldin, heldout, forward portions
        """
        batch = batch.clone() # make sure we don't corrupt the input data (which is stored in memory)

        if not contrast_mode:
            mode = self.cfg.MASK_MODE
            should_expand = self.cfg.MASK_MAX_SPAN > 1 and expand_prob > 0.0 and torch.rand(1).item() < expand_prob
            width =  torch.randint(1, self.cfg.MASK_MAX_SPAN + 1, (1, )).item() if should_expand else 1
            mask_ratio = self.cfg.MASK_RATIO if width == 1 else self.cfg.MASK_RATIO / width
        else:
            mode = self.cfg.CONTRAST_MASK_MODE
            should_expand = self.cfg.CONTRAST_MASK_MAX_SPAN > 1 and expand_prob > 0.0 and torch.rand(1).item() < expand_prob
            width =  torch.randint(1, self.cfg.CONTRAST_MASK_MAX_SPAN + 1, (1, )).item() if should_expand else 1
            mask_ratio = self.cfg.CONTRAST_MASK_RATIO if width == 1 else self.cfg.CONTRAST_MASK_RATIO / width

        labels = batch.clone()

        # batch_full includes heldin_spikes, heldout_spikes and forward_spikes, all corrupted by full mask mode
        if heldout_spikes is not None:
            batch_full = torch.cat([batch.clone(), heldout_spikes.to(batch.device)], -1)
        if forward_spikes is not None:
            batch_full = torch.cat([batch_full, forward_spikes.to(batch.device)], 1)
        labels_full = batch_full.clone()
        spikes_full = batch_full.clone()

        if mask is None:
            if self.prob_mask is None or self.prob_mask.size() != labels.size():
                if mode == "full":
                    mask_probs = torch.full(labels.shape, mask_ratio)
                    mask_probs_full = torch.full(labels_full.shape, mask_ratio)
                elif mode == "timestep":
                    single_timestep = labels[:, :, 0] # N x T
                    mask_probs = torch.full(single_timestep.shape, mask_ratio)
                    mask_probs_full = torch.full(labels_full.shape, mask_ratio)
                elif mode == "neuron":
                    single_neuron = labels[:, 0] # N x H
                    mask_probs = torch.full(single_neuron.shape, mask_ratio)
                elif mode == "timestep_only":
                    single_timestep = labels[0, :, 0] # T
                    mask_probs = torch.full(single_timestep.shape, mask_ratio)
                self.prob_mask = mask_probs.to(self.device)
                self.prob_mask_full = mask_probs_full.to(self.device)
            # If we want any tokens to not get masked, do it here (but we don't currently have any)
            mask = torch.bernoulli(self.prob_mask)
            mask_full = torch.bernoulli(self.prob_mask_full)

            if width > 1 and mode != "full":
                mask = self.expand_mask(mask, width)
                mask_full = self.expand_mask(mask_full, width)

            mask = mask.bool()
            mask_full = mask_full.bool()
            if mode == "timestep":
                mask = mask.unsqueeze(2).expand_as(labels)
            elif mode == "neuron":
                mask = mask.unsqueeze(1).expand_as(labels) # neuron dimension corrected
            elif mode == "timestep_only":
                mask = mask.unsqueeze(0).unsqueeze(2).expand_as(labels)
                # we want the shape of the mask to be T
        elif mask.size() != labels.size(): ### TODO: account for labels_full.size() also
            raise Exception(f"Input mask of size {mask.size()} does not match input size {labels.size()}")

        labels[~mask] = UNMASKED_LABEL  # No ground truth for unmasked - use this to mask loss
        labels_full[~mask_full] = UNMASKED_LABEL
        if not should_mask:
            # Only do the generation
            return batch, labels

        # We use random assignment so the model learns embeddings for non-mask tokens, and must rely on context
        # Most times, we replace tokens with MASK token
        if not contrast_mode:
            indices_replaced = torch.bernoulli(torch.full(labels.shape, self.cfg.MASK_TOKEN_RATIO, device=mask.device)).bool() & mask
            indices_replaced_full = torch.bernoulli(torch.full(labels_full.shape, self.cfg.MASK_TOKEN_RATIO, device=mask_full.device)).bool() & mask_full
        else:
            indices_replaced = torch.bernoulli(torch.full(labels.shape, self.cfg.CONTRAST_MASK_TOKEN_RATIO, device=mask.device)).bool() & mask
            indices_replaced_full = torch.bernoulli(torch.full(labels_full.shape, self.cfg.CONTRAST_MASK_TOKEN_RATIO, device=mask_full.device)).bool() & mask_full
        if self.cfg.USE_ZERO_MASK:
            batch[indices_replaced] = 0
            batch_full[indices_replaced_full] = 0
        else:
            batch[indices_replaced] = max_spikes + 1
            batch_full[indices_replaced_full] = max_spikes + 1

        # Random % of the time, we replace masked input tokens with random value (the rest are left intact)
        if not contrast_mode:
            indices_random = torch.bernoulli(torch.full(labels.shape, self.cfg.MASK_RANDOM_RATIO, device=mask.device)).bool() & mask & ~indices_replaced
            indices_random_full = torch.bernoulli(torch.full(labels_full.shape, self.cfg.MASK_RANDOM_RATIO, device=mask_full.device)).bool() & mask_full & ~indices_replaced_full
        else:
            indices_random = torch.bernoulli(torch.full(labels.shape, self.cfg.CONTRAST_MASK_RANDOM_RATIO, device=mask.device)).bool() & mask & ~indices_replaced
            indices_random_full = torch.bernoulli(torch.full(labels_full.shape, self.cfg.CONTRAST_MASK_RANDOM_RATIO, device=mask_full.device)).bool() & mask_full & ~indices_replaced_full
        random_spikes = torch.randint(batch.max(), labels.shape, dtype=torch.long, device=batch.device)
        random_spikes_full = torch.randint(batch_full.max(), labels_full.shape, dtype=torch.long, device=batch_full.device)
        batch[indices_random] = random_spikes[indices_random]
        batch_full[indices_random_full] = random_spikes_full[indices_random_full]
        masked_heldin = batch.clone()

        if heldout_spikes is not None:
            # heldout spikes are all masked
            batch = torch.cat([batch, torch.zeros_like(heldout_spikes, device=batch.device)], -1)
            labels = torch.cat([labels, heldout_spikes.to(batch.device)], -1)
        if forward_spikes is not None:
            batch = torch.cat([batch, torch.zeros_like(forward_spikes, device=batch.device)], 1)
            labels = torch.cat([labels, forward_spikes.to(batch.device)], 1)
        # Leave the other 10% alone
        return batch, labels, masked_heldin, batch_full, labels_full, spikes_full
