import torch
import torch.nn as nn
from models.encoder import EEGEncoder
from models.direct_head import DirectHead
from models.hyperbolic_prototype import HyperbolicPrototypeBranch
from models.edge_attention import EdgeAttentionBranch
from models.concept_branch import ConceptBranch


class EAGLENet(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        self.cfg = cfg
        self.graph = graph
        self.num_classes = graph['num_classes']
        self.use_direct = cfg['model'].get('use_direct_head', True)
        self.use_proto = cfg['model'].get('use_proto', True)
        self.use_edge = cfg['model'].get('use_edge', True)
        self.use_concept = cfg['model'].get('use_concept', True)
        self.direct_only = cfg['model'].get('direct_only', False)

        self.encoder = EEGEncoder(cfg)
        self.direct_head = DirectHead(cfg, self.num_classes)
        self.proto_branch = HyperbolicPrototypeBranch(cfg, graph)
        self.edge_branch = EdgeAttentionBranch(cfg, graph)
        self.concept_branch = ConceptBranch(cfg, graph)
        self.gamma_direct = cfg.get('fusion', {}).get('gamma_direct', 0.0)
        self.alpha_proto = cfg.get('fusion', {}).get('alpha_proto', 1.0)
        self.alpha_edge = cfg.get('fusion', {}).get('alpha_edge', 1.0)
        self.beta_concept = cfg.get('fusion', {}).get('beta_concept', 0.2)

    def _zero_logits(self, x):
        return x.new_zeros((x.shape[0], self.num_classes))

    def forward(self, x):
        h_tok, h_struct = self.encoder(x)
        logits_direct = self.direct_head(h_tok)
        if self.direct_only:
            N = x.shape[0]
            d_proto = self.cfg['model']['d_proto']
            T, C, B = h_struct.shape[1:4]
            zeros_proto = logits_direct.new_zeros((N, self.graph['num_nodes']))
            zeros_embed = logits_direct.new_zeros((N, d_proto))
            zeros_prototypes = logits_direct.new_zeros((self.graph['num_nodes'], d_proto))
            zeros_edge = logits_direct.new_zeros((N, self.graph['num_edges']))
            zeros_edge_attention = logits_direct.new_zeros((N, self.graph['num_edges'], T, C, B))
            zeros_edge_evidence = logits_direct.new_zeros((N, self.graph['num_edges'], h_tok.shape[-1]))
            zeros_concept = logits_direct.new_zeros((N, 3))
            return {
                'logits_final': logits_direct,
                'logits_direct': logits_direct,
                'logits_proto': self._zero_logits(logits_direct),
                'logits_edge': self._zero_logits(logits_direct),
                'logits_concept': self._zero_logits(logits_direct),
                'proto_distance': zeros_proto,
                'proto_embedding': zeros_embed,
                'proto_tangent': zeros_embed,
                'prototypes': zeros_prototypes,
                'edge_weights': zeros_edge,
                'edge_attention': zeros_edge_attention,
                'edge_evidence': zeros_edge_evidence,
                'concept_scores': zeros_concept,
                'h_tok': h_tok,
                'h_struct': h_struct,
            }
        proto = self.proto_branch(h_tok)
        edge = self.edge_branch(h_tok, token_shape=h_struct.shape[1:4])
        concept = self.concept_branch(proto['v'])

        logits_proto = proto['logits'] if self.use_proto else self._zero_logits(logits_direct)
        logits_edge = edge['logits'] if self.use_edge else self._zero_logits(logits_direct)
        logits_concept = concept['logits'] if self.use_concept else self._zero_logits(logits_direct)

        logits_final = (
            self.gamma_direct * logits_direct
            + self.alpha_proto * logits_proto
            + self.alpha_edge * logits_edge
            + self.beta_concept * logits_concept
        )
        if not (self.use_proto or self.use_edge or self.use_concept):
            logits_final = logits_direct

        return {
            'logits_final': logits_final,
            'logits_direct': logits_direct,
            'logits_proto': logits_proto,
            'logits_edge': logits_edge,
            'logits_concept': logits_concept,
            'proto_distance': proto['distance'],
            'proto_embedding': proto['z_hyperbolic'],
            'proto_tangent': proto['v'],
            'prototypes': proto['prototypes'],
            'edge_weights': edge['edge_weights'],
            'edge_attention': edge['edge_attention'],
            'edge_evidence': edge['edge_evidence'],
            'concept_scores': concept['concept_scores'],
            'h_tok': h_tok,
            'h_struct': h_struct,
        }
