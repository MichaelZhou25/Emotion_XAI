import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    """Lightweight residual convolution over EEG frequency bands."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=1)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(cfg['model'].get('dropout',0.1)), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h):
        # h: [N,T,C,B,d]
        N,T,C,B,d = h.shape
        x = h.reshape(N*T*C, B, d)
        y = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm(x + y)
        x = self.norm2(x + self.ffn(x))
        return x.reshape(N,T,C,B,d)
