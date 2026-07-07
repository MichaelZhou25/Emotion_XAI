import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Lightweight residual convolution over time windows."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=1)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(cfg['model'].get('dropout',0.1)), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h):
        # h: [N,T,C,B,d], conv over T for each channel-band
        N,T,C,B,d = h.shape
        x = h.permute(0,2,3,1,4).reshape(N*C*B, T, d)
        y = self.conv(x.transpose(1,2)).transpose(1,2)
        x = self.norm(x + y)
        x = self.norm2(x + self.ffn(x))
        return x.reshape(N,C,B,T,d).permute(0,3,1,2,4)
