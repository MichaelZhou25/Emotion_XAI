import torch
import torch.nn as nn


class HemisphericFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['model']['d_model']
        self.enabled = cfg['model'].get('use_hemi', True)
        self.num_channels = cfg['dataset'].get('num_channels', 62)
        self.pairs = cfg['model'].get('hemisphere_pairs')
        self.fuse = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def _default_pairs(self, C):
        # Conservative fallback: pair first half with reversed second half.
        half = C // 2
        return [(i, C - 1 - i) for i in range(half)]

    def forward(self, h):
        if not self.enabled:
            return h
        # h: [N,T,C,B,d]
        N,T,C,B,d = h.shape
        pairs = self.pairs or self._default_pairs(C)
        out = h.clone()
        for l, r in pairs:
            if l >= C or r >= C or l == r:
                continue
            hl = h[:,:,l,:,:]
            hr = h[:,:,r,:,:]
            evidence = torch.cat([hl - hr, torch.abs(hl - hr), hl + hr], dim=-1)
            fused = self.fuse(evidence)
            gate = self.gate(fused)
            out[:,:,l,:,:] = out[:,:,l,:,:] + gate * fused
            out[:,:,r,:,:] = out[:,:,r,:,:] + gate * fused
        return out
