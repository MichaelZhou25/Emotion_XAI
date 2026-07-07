import torch
import torch.nn as nn


class ConceptBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        dp = cfg['model']['d_proto']
        self.to_concept = nn.Sequential(nn.LayerNorm(dp), nn.Linear(dp, dp), nn.GELU(), nn.Linear(dp, 3), nn.Tanh())
        self.concept_matrix = torch.tensor(graph['concept_matrix'], dtype=torch.float32)  # [num_classes,3]

    def forward(self, v):
        scores = self.to_concept(v)
        A = self.concept_matrix.to(v.device)
        logits = torch.matmul(scores, A.t())
        return {'concept_scores': scores, 'logits': logits}
