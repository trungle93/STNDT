#!/usr/bin/env python3
# Author: Joel Ye
# Adapted by Trung Le
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer, MultiheadAttention
from torch.distributions import Poisson

from src.utils import binary_mask_to_attn_mask
from src.mask import UNMASKED_LABEL


# * Note that the TransformerEncoder and TransformerEncoderLayer were reproduced here for experimentation
# * Only minor edits were actually made to the computation.

class TransformerEncoderLayerWithHooks(TransformerEncoderLayer):
    def __init__(self, config, d_model, trial_length, device=None, **kwargs):
        super().__init__(
            d_model,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
            activation=config.ACTIVATION,
            **kwargs
        )
        self.config = config
        self.num_input = d_model
        self.trial_length = trial_length
        self.device = device
        if config.FIXUP_INIT:
            self.fixup_initialization()
        self.spatial_self_attn = MultiheadAttention(embed_dim=self.trial_length, num_heads=config.NUM_HEADS)
        self.spatial_norm1 = nn.LayerNorm(self.trial_length)

        self.ts_norm1 = nn.LayerNorm(d_model)
        self.ts_norm2 = nn.LayerNorm(d_model)
        self.ts_linear1 = nn.Linear(d_model, config.HIDDEN_SIZE)
        self.ts_linear2 = nn.Linear(config.HIDDEN_SIZE, d_model)
        self.ts_dropout1 = nn.Dropout(config.DROPOUT)
        self.ts_dropout2 = nn.Dropout(config.DROPOUT)
        self.ts_dropout3 = nn.Dropout(config.DROPOUT)

    def update_config(self, config):
        self.config = config
        self.dropout = nn.Dropout(config.DROPOUT)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.ts_dropout1 = nn.Dropout(config.DROPOUT)
        self.ts_dropout2 = nn.Dropout(config.DROPOUT)
        self.ts_dropout3 = nn.Dropout(config.DROPOUT)

    def fixup_initialization(self):
        r"""
        http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        temp_state_dic = {}
        en_layers = self.config.NUM_LAYERS

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
            elif name in ["self_attn.v_proj.weight", ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2 ** 0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def get_input_size(self):
        return self.num_input

    def attend(self, src, context_mask=None, **kwargs):
        attn_res = self.self_attn(src, src, src, attn_mask=context_mask, **kwargs)
        return (*attn_res, torch.tensor(0, device=src.device, dtype=torch.float))

    def spatial_attend(self, src, context_mask=None, **kwargs):
        attn_res = self.spatial_self_attn(src, src, src, attn_mask=context_mask, **kwargs)
        return (*attn_res, torch.tensor(0, device=src.device, dtype=torch.float))

    def forward(self, src, spatial_src, src_mask=None, spatial_src_mask=None, src_key_padding_mask=None, **kwargs):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            src: L, N, E (time x batch x neurons)
            weights: N, L, S (batch x target time x source time)
        """

        residual = src
        if self.config.PRE_NORM:
            src = self.norm1(src)

        t_out, t_weights, _ = self.attend(
            src,
            context_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = residual + self.dropout1(t_out)
        if not self.config.PRE_NORM:
            src = self.norm1(src)
        residual = src
        if self.config.PRE_NORM:
            src = self.norm2(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.config.PRE_NORM:
            src = self.norm2(src)

        spatial_residual = spatial_src
        if self.config.PRE_NORM:
            spatial_src = self.spatial_norm1(spatial_src)
        spatial_out, spatial_weights, _ = self.spatial_attend(
            spatial_src,
            context_mask=spatial_src_mask,
            key_padding_mask=None,
        )

        ts_residual = src
        if self.config.PRE_NORM:
            src = self.ts_norm1(src)
        ts_out = torch.bmm(spatial_weights, src.permute(1, 2, 0)).permute(2, 0, 1)
        ts_out = ts_residual + self.ts_dropout1(ts_out)
        if not self.config.PRE_NORM:
            ts_out = self.ts_norm1(ts_out)

        ts_residual = ts_out
        if self.config.PRE_NORM:
            ts_out = self.ts_norm2(ts_out)
        ts_out = self.ts_linear2(self.ts_dropout2(self.activation(self.ts_linear1(ts_out))))
        ts_out = ts_residual + self.ts_dropout3(ts_out)
        if not self.config.PRE_NORM:
            ts_out = self.ts_norm2(ts_out)

        return ts_out, (spatial_weights, t_weights), 0


class TransformerEncoderWithHooks(TransformerEncoder):
    r""" Hooks into transformer encoder.
    """

    def __init__(self, encoder_layer, norm=None, config=None, num_layers=None, device=None, src_pos_encoder=None,
                 spatial_pos_encoder=None):
        super().__init__(encoder_layer, config.NUM_LAYERS, norm)
        self.device = device
        self.update_config(config)

    def update_config(self, config):
        self.config = config
        for layer in self.layers:
            layer.update_config(config)

    def split_src(self, src):
        r""" More useful in inherited classes """
        return src

    def extract_return_src(self, src):
        r""" More useful in inherited classes """
        return src

    def forward(self, src, spatial_src, mask=None, spatial_mask=None, return_outputs=False, return_weights=False,
                **kwargs):
        value = src
        src = self.split_src(src)
        layer_outputs = []
        layer_weights = []
        layer_costs = []
        for i, mod in enumerate(self.layers):
            if i == 0:
                src, weights, layer_cost = mod(src, spatial_src, src_mask=mask, spatial_src_mask=spatial_mask, **kwargs)
            else:
                src, weights, layer_cost = mod(src, src.permute(2, 1, 0), src_mask=mask, spatial_src_mask=spatial_mask, **kwargs)
            if return_outputs:
                layer_outputs.append(src.permute(1,0,2)) # t x b x n -> b x t x n
            layer_weights.append(weights)
            layer_costs.append(layer_cost)
        total_layer_cost = sum(layer_costs)

        if not return_weights:
            layer_weights = None
        if not return_outputs:
            layer_outputs = None
        else:
            layer_outputs = torch.stack(layer_outputs, dim=-1)

        return_src = self.extract_return_src(src)
        if self.norm is not None:
            return_src = self.norm(return_src)

        return return_src, layer_outputs, layer_weights, total_layer_cost


class NeuralDataTransformer(nn.Module):
    r"""
        Transformer encoder-based dynamical systems decoder. Trained on MLM loss. Returns loss and predicted rates.
    """

    def __init__(self, config, trial_length, num_neurons, device, max_spikes):
        super().__init__()
        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons
        self.device = device

        # TODO buffer
        if config.FULL_CONTEXT:
            self.src_mask = None
            self.spatial_src_mask = None
        else:
            self.src_mask = {}  # multi-GPU masks
            self.spatial_src_mask = {}  # multi-GPU masks

        self.num_input = self.num_neurons
        self.num_spatial_input = self.trial_length

        assert config.EMBED_DIM == 0 or config.EMBED_DIM == 1, 'EMBED_DIM can only take values of 0 or 1'
        if config.LINEAR_EMBEDDER:
            self.embedder = nn.Sequential(nn.Linear(self.num_neurons, self.num_input))
            self.spatial_embedder = nn.Sequential(nn.Linear(self.trial_length, self.num_spatial_input))
        elif config.EMBED_DIM == 0:
            self.embedder = nn.Identity()
            self.spatial_embedder = nn.Identity()
        else: # config.EMBED_DIM == 1
            self.embedder = nn.Sequential(
                nn.Embedding(max_spikes + 2, config.EMBED_DIM),
                nn.Flatten(start_dim=-2)
            )
            self.spatial_embedder = nn.Sequential(
                nn.Embedding(max_spikes + 2, config.EMBED_DIM),
                nn.Flatten(start_dim=-2)
            )

        self.scale = math.sqrt(self.get_factor_size())
        self.spatial_scale = math.sqrt(self.get_factor_size(spatial=True))
        self.src_pos_encoder = PositionalEncoding(config, self.trial_length, self.get_factor_size(), device)
        self.spatial_pos_encoder = PositionalEncoding(config, self.get_factor_size(), self.trial_length, device)

        if config.USE_CONTRAST_PROJECTOR:
            if config.LINEAR_PROJECTOR:
                self.projector = nn.Linear(self.get_factor_size(), self.get_factor_size())
                self.spatial_projector = nn.Linear(self.get_factor_size(spatial=True), self.get_factor_size(spatial=True))
            else:
                self.projector = nn.Sequential(nn.Linear(self.get_factor_size(), 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024,self.get_factor_size()),
                                            )
                self.spatial_projector = nn.Sequential(nn.Linear(self.get_factor_size(spatial=True), 1024).
                                                    nn.ReLU(),
                                                    nn.Linear(1024, self.get_factor_size(spatial=True)))

        else:
            self.projector = nn.Identity()
            self.spatial_projector = nn.Identity()
        self.n_views = 2
        self.temperature = self.config.TEMPERATURE
        self.contrast_lambda = self.config.LAMBDA
        self.cel = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='mean')

        self._init_transformer()

        self.rate_dropout = nn.Dropout(config.DROPOUT_RATES)

        if config.LOSS.TYPE == "poisson":
            src_decoder_layers = []
            if config.DECODER.LAYERS == 1:
                src_decoder_layers.append(nn.Linear(self.get_factor_size(), self.num_input))
            else:
                decoder_layers.append(nn.Linear(self.get_factor_size(), 16))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(16, self.num_neurons))
            if not config.LOGRATE:
                decoder_layers.append(nn.ReLU())  # If we're not using lograte, we need to feed positive rates
            self.src_decoder = nn.Sequential(*src_decoder_layers)
            self.classifier = nn.PoissonNLLLoss(reduction='none', log_input=config.LOGRATE)
        elif config.LOSS.TYPE == "cel":  # Note - we need a different spike count mechanism
            self.decoder = nn.Sequential(
                nn.Linear(self.get_factor_size(), config.MAX_SPIKE_COUNT * self.num_neurons)  # log-likelihood
            )
            self.classifier = nn.CrossEntropyLoss(reduction='none')
        else:
            raise Exception(f"{config.LOSS.TYPE} loss not implemented")

        self.init_weights()

    def update_config(self, config):
        r"""
            Update config -- currently, just replaces config and replaces the dropout layers
        """
        self.config = config
        self.rate_dropout = nn.Dropout(config.DROPOUT_RATES)
        self.src_pos_encoder.update_config(config)
        self.transformer_encoder.update_config(config)
        self.src_mask = {}  # Clear cache
        self.spatial_src_mask = {}  # Clear cache

    def get_factor_size(self, spatial=False):
        if spatial:
            return self.num_spatial_input
        else:
            return self.num_input

    def get_hidden_size(self):
        return self.num_input
        
    def get_encoder_layer(self):
        return TransformerEncoderLayerWithHooks

    def get_encoder(self):
        return TransformerEncoderWithHooks

    def _init_transformer(self):
        assert issubclass(self.get_encoder_layer(), TransformerEncoderLayerWithHooks)
        assert issubclass(self.get_encoder(), TransformerEncoderWithHooks)
        encoder_layer = self.get_encoder_layer()(self.config, d_model=self.get_factor_size(), trial_length=self.trial_length, device=self.device)
        if self.config.SCALE_NORM:
            norm = ScaleNorm(self.get_factor_size() ** 0.5)
        else:
            norm = nn.LayerNorm(self.get_factor_size())
        self.transformer_encoder = self.get_encoder()(
            encoder_layer,
            norm=norm,
            config=self.config,
            device=self.device,
        )

    def _get_or_generate_context_mask(self, src, do_convert=True, expose_ic=True, spatial=False):
        if self.config.FULL_CONTEXT:
            return None
        if str(src.device) in self.spatial_src_mask and spatial:
            return self.spatial_src_mask[str(src.device)]
        if str(src.device) in self.src_mask and not spatial:
            return self.src_mask[str(src.device)]
        size = src.size(0)
        context_forward = self.config.CONTEXT_FORWARD
        if self.config.CONTEXT_FORWARD < 0:
            context_forward = size
        mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1).transpose(0, 1)
        if self.config.CONTEXT_BACKWARD > 0:
            back_mask = (torch.triu(torch.ones(size, size, device=src.device),
                                    diagonal=-self.config.CONTEXT_BACKWARD) == 1)
            mask = mask & back_mask
        if expose_ic and self.config.CONTEXT_WRAP_INITIAL and self.config.CONTEXT_BACKWARD > 0:
            # Expose initial segment for IC
            initial_mask = torch.triu(
                torch.ones(self.config.CONTEXT_BACKWARD, self.config.CONTEXT_BACKWARD, device=src.device))
            mask[:self.config.CONTEXT_BACKWARD, :self.config.CONTEXT_BACKWARD] |= initial_mask
        mask = mask.float()
        if do_convert:
            mask = binary_mask_to_attn_mask(mask)
        if spatial:
            self.spatial_src_mask[str(src.device)] = mask
        else:
            self.src_mask[str(src.device)] = mask
        return self.src_mask[str(src.device)] if not spatial else self.spatial_src_mask[str(src.device)]

    def init_weights(self):
        r"""
            Init hoping for better optimization.
            Sources:
            Transformers without Tears https://arxiv.org/pdf/1910.05895.pdf
            T-Fixup http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        initrange = 0.1
        if self.config.EMBED_DIM != 0:
            if hasattr(self.config, "SPIKE_LOG_INIT") and self.config.SPIKE_LOG_INIT:
                # Use a log scale, since we expect spike semantics to follow compressive distribution
                max_spikes = self.embedder[0].num_embeddings + 1
                log_scale = torch.arange(1, max_spikes).float().log()  # 1 to lg
                log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                log_scale = log_scale * initrange
                # Add some noise
                self.embedder[0].weight.data.uniform_(-initrange / 10, initrange / 10)
                self.embedder[0].weight.data += log_scale.unsqueeze(1).expand_as(self.embedder[0].weight.data)
                self.spatial_embedder[0].weight.data.uniform_(-initrange / 10, initrange / 10)
                self.spatial_embedder[0].weight.data += log_scale.unsqueeze(1).expand_as(self.embedder[0].weight.data)
            else:
                self.embedder[0].weight.data.uniform_(-initrange, initrange)

        self.src_decoder[0].bias.data.zero_()
        self.src_decoder[0].weight.data.uniform_(-initrange, initrange)

    # adapted from https://github.com/sthalles/SimCLR.git
    def info_nce_loss(self, features):
        batch_size = features.shape[0] / 2
        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, src, mask_labels, contrast_src1=None, contrast_src2=None, val_phase=False, **kwargs):
        src = src.float()
        if contrast_src1 is not None and contrast_src2 is not None:
            contrast_src1 = contrast_src1.float()
            contrast_src2 = contrast_src2.float()

        spatial_src = src.permute(2,0,1)
        spatial_src = self.spatial_embedder(spatial_src) * self.spatial_scale
        spatial_src = self.spatial_pos_encoder(spatial_src)
        src = src.permute(1,0,2)
        src = self.embedder(src) * self.scale
        src = self.src_pos_encoder(src)
        src_mask = self._get_or_generate_context_mask(src)
        spatial_src_mask = None
        (
            encoder_output,
            layer_outputs,
            layer_weights,
            _
        ) = self.transformer_encoder(src, spatial_src, src_mask, spatial_src_mask, **kwargs)
        encoder_output = self.rate_dropout(encoder_output)
        decoder_output = self.src_decoder(encoder_output)

        if contrast_src1 is not None and contrast_src2 is not None:
            spatial_contrast1 = contrast_src1.permute(2,0,1)
            spatial_contrast_embedded1 = self.spatial_embedder(spatial_contrast1) * self.spatial_scale
            spatial_contrast1 = self.spatial_pos_encoder(spatial_contrast_embedded1)
            
            contrast_src1 = contrast_src1.permute(1,0,2)
            contrast_src_embedded1 = self.embedder(contrast_src1) * self.scale
            contrast_src1 = self.src_pos_encoder(contrast_src_embedded1)
            contrast_mask1 = self._get_or_generate_context_mask(contrast_src1)
            spatial_contrast_mask1 = None
            (
                encoder_output_contrast1,
                layer_outputs_contrast1,
                _,_
            ) = self.transformer_encoder(contrast_src1, spatial_contrast1, contrast_mask1, spatial_contrast_mask1, **kwargs)
            encoder_output_contrast1 = self.rate_dropout(encoder_output_contrast1)

            spatial_contrast2 = contrast_src2.permute(2,0,1)
            spatial_contrast_embedded2 = self.spatial_embedder(spatial_contrast2) * self.spatial_scale
            spatial_contrast2 = self.spatial_pos_encoder(spatial_contrast_embedded2)
            
            contrast_src2 = contrast_src2.permute(1,0,2)
            contrast_src_embedded2 = self.embedder(contrast_src2) * self.scale
            contrast_src2 = self.src_pos_encoder(contrast_src_embedded2)
            contrast_mask2 = self._get_or_generate_context_mask(contrast_src2)
            spatial_contrast_mask2 = None
            (
                encoder_output_contrast2,
                layer_outputs_contrast2,
                _,_
            ) = self.transformer_encoder(contrast_src2, spatial_contrast2, contrast_mask2, spatial_contrast_mask2, **kwargs)
            encoder_output_contrast2 = self.rate_dropout(encoder_output_contrast2)

            decoder_output_contrast1 = self.src_decoder(encoder_output_contrast1)
            decoder_output_contrast2 = self.src_decoder(encoder_output_contrast2)

            if self.config.CONTRAST_LAYER == 'embedder':
                out1 = contrast_src_embedded1
                out2 = contrast_src_embedded2
            elif self.config.CONTRAST_LAYER == 'decoder':
                out1 = decoder_output_contrast1
                out2 = decoder_output_contrast2
            else:
                out1 = layer_outputs_contrast1[self.config.CONTRAST_LAYER]
                out2 = layer_outputs_contrast2[self.config.CONTRAST_LAYER]
            out1 = torch.flatten(self.projector(out1.permute(1,0,2)), start_dim=1, end_dim=2)
            out2 = torch.flatten(self.projector(out2.permute(1,0,2)), start_dim=1, end_dim=2)
            features = torch.cat([out1, out2], dim=0)
            logits, labels = self.info_nce_loss(features)
            if not val_phase:
                contrast_loss = self.cel(logits, labels).mean() * self.contrast_lambda
            else:
                contrast_loss = self.cel(logits, labels) * self.contrast_lambda

            if self.config.CONTRAST_LAYER == 'embedder':
                out1 = spatial_contrast_embedded1
                out2 = spatial_contrast_embedded2
                out1 = torch.flatten(self.spatial_projector(out1.permute(1,0,2)), start_dim=1, end_dim=2)
                out2 = torch.flatten(self.spatial_projector(out2.permute(1,0,2)), start_dim=1, end_dim=2)
                features = torch.cat([out1, out2], dim=0)
                logits, labels = self.info_nce_loss(features)
                if not val_phase:
                    contrast_loss = contrast_loss + self.cel(logits, labels).mean() * self.contrast_lambda
                else:
                    contrast_loss = contrast_loss + self.cel(logits, labels) * self.contrast_lambda
        else:
            contrast_loss = torch.tensor([0.0]).to(self.device)

        if self.config.LOSS.TYPE == "poisson":
            decoder_rates = decoder_output.permute(1, 0, 2)
            decoder_loss = self.classifier(decoder_rates, mask_labels)
        masked_decoder_loss = decoder_loss[mask_labels != UNMASKED_LABEL]
        if not val_phase:
            masked_decoder_loss = masked_decoder_loss.mean()
            loss = masked_decoder_loss + contrast_loss
        else:
            loss = masked_decoder_loss
        if not val_phase:
            return (
                loss.unsqueeze(0),
                masked_decoder_loss.unsqueeze(0),
                contrast_loss.unsqueeze(0),
                decoder_rates,
            )
        else:
            return (
                loss,
                masked_decoder_loss,
                contrast_loss,
                decoder_rates,
                layer_weights,
                layer_outputs,
            )


class PositionalEncoding(nn.Module):
    r"""
    ! FYI - needs even d_model if not learned.
    """

    def __init__(self, cfg, trial_length, d_model, device):
        super().__init__()
        self.dropout = nn.Dropout(p=cfg.DROPOUT_EMBEDDING)
        pe = torch.zeros(trial_length, d_model).to(device)  # * Can optim to empty
        position = torch.arange(0, trial_length, dtype=torch.float).unsqueeze(1)
        if cfg.POSITION.OFFSET:
            position = position + 1
        self.learnable = cfg.LEARNABLE_POSITION
        if self.learnable:
            self.register_buffer('pe', position.long())
            self.pos_embedding = nn.Embedding(trial_length, d_model)  # So maybe it's here...?
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

    def update_config(self, config):
        self.dropout = nn.Dropout(config.DROPOUT_EMBEDDING)

    def forward(self, x):
        if self.learnable:
            x = x + self.pos_embedding(self.pe)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


