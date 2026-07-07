import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h_tok):
        a = torch.softmax(self.score(h_tok).squeeze(-1), dim=-1)
        return torch.einsum('nl,nld->nd', a, h_tok)


class DirectHead(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        d = cfg['model']['d_model']
        self.pool = AttentionPool(d)
        self.cls = nn.Sequential(nn.LayerNorm(d), nn.Dropout(cfg['model'].get('dropout',0.1)), nn.Linear(d, num_classes))

    def forward(self, h_tok):
        return self.cls(self.pool(h_tok))
