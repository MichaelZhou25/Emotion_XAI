import math
import torch
import torch.nn as nn


class EdgeAttentionBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        d = cfg['model']['d_model']
        self.num_edges = graph['num_edges']
        self.T = cfg['dataset']['time_steps']
        self.C = cfg['dataset']['num_channels']
        self.B = cfg['dataset']['num_bands']
        self.edge_queries = nn.Parameter(torch.randn(self.num_edges, d) * 0.02)
        self.edge_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.path_matrix = torch.tensor(graph['path_matrix'], dtype=torch.float32)  # [num_classes,E]

    def forward(self, h_tok, token_shape=None):
        # h_tok: [N,L,d]
        N,L,d = h_tok.shape
        q = self.edge_queries.unsqueeze(0).expand(N, -1, -1)
        score = torch.matmul(q, h_tok.transpose(1,2)) / math.sqrt(d)
        attn = torch.softmax(score, dim=-1)  # [N,E,L]
        edge_evidence = torch.matmul(attn, h_tok)  # [N,E,d]
        edge_weights = torch.sigmoid(self.edge_mlp(edge_evidence).squeeze(-1))
        M = self.path_matrix.to(h_tok.device)
        logits = torch.matmul(edge_weights, M.t())
        T, C, B = token_shape or (self.T, self.C, self.B)
        if L != T * C * B:
            raise ValueError(f'Cannot reshape edge attention length {L} to [T,C,B]=[{T},{C},{B}]')
        edge_attention = attn.reshape(N, self.num_edges, T, C, B)
        return {'logits': logits, 'edge_weights': edge_weights, 'edge_attention': edge_attention, 'edge_evidence': edge_evidence}
