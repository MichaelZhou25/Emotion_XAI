import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """Lightweight residual convolution over channel dimension."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']
        self.conv = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=1)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(cfg['model'].get('dropout',0.1)), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h):
        # h: [N,T,C,B,d], conv over C for each N,T,B
        N,T,C,B,d = h.shape
        x = h.permute(0,1,3,2,4).reshape(N*T*B, C, d)
        y = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm(x + y)
        x = self.norm2(x + self.ffn(x))
        return x.reshape(N,T,B,C,d).permute(0,1,3,2,4)
