import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.direct_head import AttentionPool


def project_ball(x, eps=1e-5):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    maxnorm = 1.0 - eps
    factor = torch.clamp(maxnorm / norm, max=1.0)
    return x * factor


def expmap0(v, eps=1e-8):
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return project_ball(torch.tanh(norm) * v / norm)


def poincare_distance(x, y, eps=1e-5):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(0)
    diff2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1)
    denom = (1 - x2).clamp_min(eps) * (1 - y2).clamp_min(eps)
    z = 1 + 2 * diff2 / denom
    z = torch.clamp(z, min=1 + eps)
    return torch.acosh(z)


class HyperbolicPrototypeBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        d = cfg['model']['d_model']
        dp = cfg['model']['d_proto']
        self.pool = AttentionPool(d)
        self.project = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, dp))
        self.temperature = cfg['model'].get('proto_temperature', 0.5)
        self.num_nodes = graph['num_nodes']
        self.class_node_indices = torch.tensor(graph.get('class_node_indices', list(range(graph['num_classes']))), dtype=torch.long)
        coords = torch.tensor(graph['semantic_coords'], dtype=torch.float32)
        self.coord_proj = nn.Linear(3, dp, bias=False)
        nn.init.xavier_uniform_(self.coord_proj.weight)
        depth = torch.tensor(graph['node_depth'], dtype=torch.float32).unsqueeze(-1)
        radius = 0.08 + 0.18 * depth
        with torch.no_grad():
            init = self.coord_proj(coords)
            init = F.normalize(init, dim=-1) * radius
        self.prototype_tangent = nn.Parameter(init)

    def prototypes(self):
        return expmap0(self.prototype_tangent)

    def forward(self, h_tok):
        r = self.pool(h_tok)
        v = self.project(r)
        z = expmap0(v)
        p = self.prototypes()
        dist = poincare_distance(z, p)
        logits_all = - dist.pow(2) / self.temperature
        class_indices = self.class_node_indices.to(h_tok.device)
        logits = logits_all.index_select(1, class_indices)
        return {'logits': logits, 'distance': dist, 'z_hyperbolic': z, 'v': v, 'prototypes': p}
