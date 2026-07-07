import torch
import torch.nn as nn
from models.spectral_encoder import SpectralEncoder
from models.spatial_encoder import SpatialEncoder
from models.temporal_encoder import TemporalEncoder
from models.hemispheric_fusion import HemisphericFusion


class EEGEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']
        self.input_mode = cfg['model'].get('input_mode', 'scalar_tokens')
        self.pre_temporal_pool = cfg['model'].get('pre_temporal_pool', 'none')
        self.num_channels = cfg['dataset'].get('num_channels', 62)
        self.num_bands = cfg['dataset'].get('num_bands', 5)
        if self.input_mode == 'feature_sequence':
            self.feature_embedding = nn.Sequential(nn.Linear(self.num_channels * self.num_bands, d), nn.LayerNorm(d))
        elif self.input_mode == 'scalar_tokens':
            self.token_embedding = nn.Sequential(nn.Linear(1, d), nn.LayerNorm(d))
            self.spectral = SpectralEncoder(cfg)
            self.spatial = SpatialEncoder(cfg)
            self.hemi = HemisphericFusion(cfg)
        else:
            raise ValueError(f'Unsupported input_mode: {self.input_mode}')
        self.temporal = TemporalEncoder(cfg)

    def forward(self, x):
        # x: [N,T,C,B]
        if self.pre_temporal_pool == 'mean':
            x = x.mean(dim=1, keepdim=True)
        elif self.pre_temporal_pool not in ('none', None):
            raise ValueError(f'Unsupported pre_temporal_pool: {self.pre_temporal_pool}')
        if self.input_mode == 'feature_sequence':
            N, T, C, B = x.shape
            h = self.feature_embedding(x.reshape(N, T, C * B)).unsqueeze(2).unsqueeze(3)
            h = self.temporal(h)
        else:
            h = self.token_embedding(x.unsqueeze(-1))
            h = self.spectral(h)
            h = self.spatial(h)
            h = self.temporal(h)
            h = self.hemi(h)
        N,T,C,B,d = h.shape
        h_tok = h.reshape(N, T*C*B, d)
        return h_tok, h
