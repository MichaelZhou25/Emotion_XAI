import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.hyperbolic_prototype import expmap0, poincare_distance
from models.temporal_encoder import TemporalEncoder


DEFAULT_HEMI_PAIRS = [
    (0, 2),    # FP1-FP2
    (3, 4),    # AF3-AF4
    (5, 13),   # F7-F8
    (6, 12),   # F5-F6
    (7, 11),   # F3-F4
    (8, 10),   # F1-F2
    (14, 22),  # FT7-FT8
    (15, 21),  # FC5-FC6
    (16, 20),  # FC3-FC4
    (17, 19),  # FC1-FC2
    (23, 31),  # T7-T8
    (24, 30),  # C5-C6
    (25, 29),  # C3-C4
    (26, 28),  # C1-C2
    (32, 40),  # TP7-TP8
    (33, 39),  # CP5-CP6
    (34, 38),  # CP3-CP4
    (35, 37),  # CP1-CP2
    (41, 49),  # P7-P8
    (42, 48),  # P5-P6
    (43, 47),  # P3-P4
    (44, 46),  # P1-P2
    (50, 56),  # PO7-PO8
    (51, 55),  # PO5-PO6
    (52, 54),  # PO3-PO4
    (57, 61),  # CB1-CB2
    (58, 60),  # O1-O2
]

DEFAULT_REGION_GROUPS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 14],
    [11, 12, 13, 15, 16, 17],
    [18, 19, 20, 21],
    [22, 23, 24, 25, 26],
]


class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.einsum('bl,bld->bd', weights, tokens)
        return pooled, weights


class TokenEncoder(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, num_layers=1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)

    def set_lambda(self, lambd):
        self.lambd = float(lambd)


class SubjectDiscriminator(nn.Module):
    def __init__(self, d_model, num_subjects, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_subjects),
        )

    def forward(self, z):
        return self.net(z)


class HemisphericEvidenceFusion(nn.Module):
    def __init__(self, d_model, hemi_pairs=None):
        super().__init__()
        self.hemi_pairs = hemi_pairs or DEFAULT_HEMI_PAIRS
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pool = AttnPool(d_model)

    def forward(self, h_channel):
        left_idx = torch.tensor([p[0] for p in self.hemi_pairs], dtype=torch.long, device=h_channel.device)
        right_idx = torch.tensor([p[1] for p in self.hemi_pairs], dtype=torch.long, device=h_channel.device)
        h_left = h_channel.index_select(1, left_idx)
        h_right = h_channel.index_select(1, right_idx)
        diff = h_left - h_right
        asym = torch.abs(diff)
        summ = h_left + h_right
        h_hemi = self.fuse(torch.cat([diff, asym, summ], dim=-1))
        _, pair_attention = self.pool(h_hemi)
        return h_hemi, pair_attention


class RegionEvidenceAggregation(nn.Module):
    def __init__(self, d_model, region_groups=None):
        super().__init__()
        self.region_groups = region_groups or DEFAULT_REGION_GROUPS
        self.pool = AttnPool(d_model)

    def forward(self, h_hemi):
        region_tokens = []
        for group in self.region_groups:
            idx = torch.tensor(group, dtype=torch.long, device=h_hemi.device)
            idx = idx[idx < h_hemi.shape[1]]
            if idx.numel() == 0:
                region_tokens.append(h_hemi.new_zeros(h_hemi.shape[0], h_hemi.shape[-1]))
            else:
                region_tokens.append(h_hemi.index_select(1, idx).mean(dim=1))
        h_region = torch.stack(region_tokens, dim=1)
        _, region_attention = self.pool(h_region)
        return h_region, region_attention


class VectorDirectBranch(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, z):
        return self.cls(z)


class VectorHyperbolicPrototypeBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        d_model = cfg['model']['d_model']
        d_proto = cfg['model'].get('d_proto', 32)
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_proto))
        self.temperature = cfg['model'].get('proto_temperature', 0.5)
        self.class_node_indices = torch.tensor(
            graph.get('class_node_indices', list(range(graph['num_classes']))),
            dtype=torch.long,
        )
        coords = torch.tensor(graph['semantic_coords'], dtype=torch.float32)
        self.coord_proj = nn.Linear(3, d_proto, bias=False)
        nn.init.xavier_uniform_(self.coord_proj.weight)
        depth = torch.tensor(graph['node_depth'], dtype=torch.float32).unsqueeze(-1)
        radius = 0.08 + 0.18 * depth
        with torch.no_grad():
            init = self.coord_proj(coords)
            init = F.normalize(init, dim=-1) * radius
        self.prototype_tangent = nn.Parameter(init)

    def prototypes(self):
        return expmap0(self.prototype_tangent)

    def forward(self, z_fused):
        v = self.project(z_fused)
        z_hyp = expmap0(v)
        prototypes = self.prototypes()
        distance = poincare_distance(z_hyp, prototypes)
        logits_all = -distance.pow(2) / self.temperature
        class_indices = self.class_node_indices.to(z_fused.device)
        logits = logits_all.index_select(1, class_indices)
        return {
            'logits': logits,
            'distance': distance,
            'z_hyperbolic': z_hyp,
            'v': v,
            'prototypes': prototypes,
        }


class MultiViewEdgeBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        d_model = cfg['model']['d_model']
        self.num_edges = graph['num_edges']
        self.logit_mode = cfg['model'].get('edge_logit_mode', 'path')
        if self.logit_mode not in {'path', 'neutral_evidence'}:
            raise ValueError(f'Unknown edge_logit_mode: {self.logit_mode}')
        self.edge_queries = nn.Parameter(torch.randn(self.num_edges, d_model) * 0.02)
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.path_matrix = torch.tensor(graph['path_matrix'], dtype=torch.float32)

    def forward(self, h_time, h_freq, h_hemi, h_region):
        tokens = torch.cat([h_time, h_freq, h_hemi, h_region], dim=1)
        n_time = h_time.shape[1]
        n_freq = h_freq.shape[1]
        n_pair = h_hemi.shape[1]
        n_region = h_region.shape[1]
        n_batch, _, d_model = tokens.shape

        q = self.edge_queries.unsqueeze(0).expand(n_batch, -1, -1)
        score = torch.matmul(q, tokens.transpose(1, 2)) / math.sqrt(d_model)
        attn = torch.softmax(score, dim=-1)
        evidence = torch.matmul(attn, tokens)
        edge_weights = torch.sigmoid(self.edge_mlp(evidence).squeeze(-1))
        path_matrix = self.path_matrix.to(tokens.device)
        logits = torch.matmul(edge_weights, path_matrix.t())
        neutral_evidence = torch.prod(1.0 - edge_weights, dim=1)
        if self.logit_mode == 'neutral_evidence':
            root_class_mask = path_matrix.abs().sum(dim=1).eq(0).to(logits.dtype)
            logits = logits + neutral_evidence.unsqueeze(1) * root_class_mask.unsqueeze(0)

        start = 0
        edge_time_attention = attn[:, :, start:start + n_time]
        start += n_time
        edge_freq_attention = attn[:, :, start:start + n_freq]
        start += n_freq
        edge_pair_attention = attn[:, :, start:start + n_pair]
        start += n_pair
        edge_region_attention = attn[:, :, start:start + n_region]

        return {
            'logits': logits,
            'edge_weights': edge_weights,
            'edge_attention': attn,
            'edge_evidence': evidence,
            'neutral_evidence': neutral_evidence,
            'edge_time_attention': edge_time_attention,
            'edge_freq_attention': edge_freq_attention,
            'edge_pair_attention': edge_pair_attention,
            'edge_region_attention': edge_region_attention,
        }


class MultiViewConceptBranch(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        d_model = cfg['model']['d_model']
        num_concepts = cfg['model'].get('num_concepts', 12)
        self.to_concepts = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_concepts),
            nn.Tanh(),
        )
        self.cls = nn.Linear(num_concepts, num_classes)

    def forward(self, z_freq, z_hemi, z_region):
        concept_input = torch.cat([z_freq, z_hemi, z_region], dim=-1)
        concept_scores = self.to_concepts(concept_input)
        logits = self.cls(concept_scores)
        return {'concept_scores': concept_scores, 'logits': logits}


class HemiMVEAGLENet(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        self.cfg = cfg
        self.graph = graph
        model_cfg = cfg['model']
        dataset_cfg = cfg['dataset']
        d_model = model_cfg['d_model']
        dropout = model_cfg.get('dropout', 0.1)
        num_heads = model_cfg.get('num_heads', 4)
        token_layers = model_cfg.get('token_encoder_layers', 1)

        self.num_classes = graph['num_classes']
        self.window_size = model_cfg.get('window_size', dataset_cfg.get('time_steps', 30))
        self.num_channels = model_cfg.get('num_channels', dataset_cfg.get('num_channels', 62))
        self.num_bands = model_cfg.get('num_bands', dataset_cfg.get('num_bands', 5))

        self.use_time_view = model_cfg.get('use_time_view', True)
        self.use_freq_view = model_cfg.get('use_freq_view', True)
        self.use_channel_view = model_cfg.get('use_channel_view', True)
        self.use_hemi_fusion = model_cfg.get('use_hemi_fusion', True)
        self.use_region_aggregation = model_cfg.get('use_region_aggregation', True)
        self.use_view_identity_embeddings = model_cfg.get('use_view_identity_embeddings', False)
        self.use_direct = model_cfg.get('use_direct_head', True)
        self.use_proto = model_cfg.get('use_proto', True)
        self.use_edge = model_cfg.get('use_edge', True)
        self.use_concept = model_cfg.get('use_concept', True)
        self.num_concepts = model_cfg.get('num_concepts', 12)
        self.use_domain_adversarial = model_cfg.get('use_domain_adversarial', False)
        self.num_domains = model_cfg.get('num_domains', dataset_cfg.get('num_subjects', 15))

        self.time_embed = nn.Sequential(nn.Linear(self.num_channels * self.num_bands, d_model), nn.LayerNorm(d_model))
        self.time_encoder = TemporalEncoder(cfg)
        self.freq_embed = nn.Sequential(nn.Linear(self.window_size * self.num_channels, d_model), nn.LayerNorm(d_model))
        self.freq_encoder = TokenEncoder(d_model, num_heads=num_heads, dropout=dropout, num_layers=token_layers)
        self.channel_embed = nn.Sequential(nn.Linear(self.window_size * self.num_bands, d_model), nn.LayerNorm(d_model))
        self.channel_encoder = TokenEncoder(d_model, num_heads=num_heads, dropout=dropout, num_layers=token_layers)
        if self.use_view_identity_embeddings:
            self.freq_identity = nn.Parameter(torch.empty(1, self.num_bands, d_model))
            self.channel_identity = nn.Parameter(torch.empty(1, self.num_channels, d_model))
            nn.init.normal_(self.freq_identity, std=0.02)
            nn.init.normal_(self.channel_identity, std=0.02)
        else:
            self.register_parameter('freq_identity', None)
            self.register_parameter('channel_identity', None)

        self.hemi_fusion = HemisphericEvidenceFusion(d_model, model_cfg.get('hemi_pairs'))
        self.region_aggregation = RegionEvidenceAggregation(d_model, model_cfg.get('region_groups'))
        self.num_hemi_pairs = len(self.hemi_fusion.hemi_pairs)
        self.num_regions = len(self.region_aggregation.region_groups)

        self.time_pool = AttnPool(d_model)
        self.freq_pool = AttnPool(d_model)
        self.channel_pool = AttnPool(d_model)
        self.region_pool = AttnPool(d_model)
        self.view_gate = nn.Linear(d_model, 1)

        self.direct_branch = VectorDirectBranch(d_model, self.num_classes, dropout=dropout)
        self.proto_branch = VectorHyperbolicPrototypeBranch(cfg, graph)
        self.edge_branch = MultiViewEdgeBranch(cfg, graph)
        self.concept_branch = MultiViewConceptBranch(cfg, self.num_classes)
        self.grl = GradientReversal(model_cfg.get('grl_lambda', 0.2))
        self.subject_discriminator = SubjectDiscriminator(d_model, self.num_domains, dropout=dropout)

        weights = model_cfg.get('branch_weights', {})
        self.w_direct = weights.get('direct', 1.0)
        self.w_proto = weights.get('proto', 0.3)
        self.w_edge = weights.get('edge', 0.2)
        self.w_concept = weights.get('concept', 0.1)

    def set_grl_lambda(self, lambd):
        self.grl.set_lambda(lambd)

    def _time_view(self, x):
        n_batch, n_time, n_channel, n_band = x.shape
        h = self.time_embed(x.reshape(n_batch, n_time, n_channel * n_band))
        h = self.time_encoder(h.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
        return h

    def _freq_view(self, x):
        n_batch = x.shape[0]
        h = x.permute(0, 3, 1, 2).reshape(n_batch, self.num_bands, self.window_size * self.num_channels)
        h = self.freq_embed(h)
        if self.freq_identity is not None:
            h = h + self.freq_identity
        return self.freq_encoder(h)

    def _channel_view(self, x):
        n_batch = x.shape[0]
        h = x.permute(0, 2, 1, 3).reshape(n_batch, self.num_channels, self.window_size * self.num_bands)
        h = self.channel_embed(h)
        if self.channel_identity is not None:
            h = h + self.channel_identity
        return self.channel_encoder(h)

    @staticmethod
    def _weighted_pool(tokens, weights):
        return torch.einsum('bl,bld->bd', weights, tokens)

    def _zero_proto(self, x):
        d_proto = self.cfg['model'].get('d_proto', 32)
        return {
            'logits': x.new_zeros((x.shape[0], self.num_classes)),
            'distance': x.new_zeros((x.shape[0], self.graph['num_nodes'])),
            'z_hyperbolic': x.new_zeros((x.shape[0], d_proto)),
            'v': x.new_zeros((x.shape[0], d_proto)),
            'prototypes': x.new_zeros((self.graph['num_nodes'], d_proto)),
        }

    def _zero_edge(self, h_time, h_freq, h_hemi, h_region):
        n_batch = h_time.shape[0]
        d_model = h_time.shape[-1]
        n_time = h_time.shape[1]
        n_freq = h_freq.shape[1]
        n_pair = h_hemi.shape[1]
        n_region = h_region.shape[1]
        n_tokens = n_time + n_freq + n_pair + n_region
        return {
            'logits': h_time.new_zeros((n_batch, self.num_classes)),
            'edge_weights': h_time.new_zeros((n_batch, self.graph['num_edges'])),
            'edge_attention': h_time.new_zeros((n_batch, self.graph['num_edges'], n_tokens)),
            'edge_evidence': h_time.new_zeros((n_batch, self.graph['num_edges'], d_model)),
            'neutral_evidence': h_time.new_zeros((n_batch,)),
            'edge_time_attention': h_time.new_zeros((n_batch, self.graph['num_edges'], n_time)),
            'edge_freq_attention': h_time.new_zeros((n_batch, self.graph['num_edges'], n_freq)),
            'edge_pair_attention': h_time.new_zeros((n_batch, self.graph['num_edges'], n_pair)),
            'edge_region_attention': h_time.new_zeros((n_batch, self.graph['num_edges'], n_region)),
        }

    def _zero_concept(self, x):
        return {
            'concept_scores': x.new_zeros((x.shape[0], self.num_concepts)),
            'logits': x.new_zeros((x.shape[0], self.num_classes)),
        }

    def _forward_edge(self, h_time, h_freq, h_hemi, h_region, proto):
        if self.use_edge:
            return self.edge_branch(h_time, h_freq, h_hemi, h_region)
        return self._zero_edge(h_time, h_freq, h_hemi, h_region)

    def _combine_logits(self, logits_direct, logits_proto, logits_edge, logits_concept):
        return (
            self.w_direct * logits_direct
            + self.w_proto * logits_proto
            + self.w_edge * logits_edge
            + self.w_concept * logits_concept
        )

    def forward(self, x):
        n_batch = x.shape[0]
        d_model = self.cfg['model']['d_model']
        h_time = self._time_view(x) if self.use_time_view else x.new_zeros((n_batch, self.window_size, d_model))
        h_freq = self._freq_view(x) if self.use_freq_view else x.new_zeros((n_batch, self.num_bands, d_model))
        h_channel = self._channel_view(x) if self.use_channel_view else x.new_zeros((n_batch, self.num_channels, d_model))

        if self.use_hemi_fusion and self.use_channel_view:
            h_hemi, pair_attention = self.hemi_fusion(h_channel)
        else:
            h_hemi = x.new_zeros((n_batch, self.num_hemi_pairs, d_model))
            pair_attention = x.new_zeros((n_batch, self.num_hemi_pairs))
        if self.use_region_aggregation and self.use_hemi_fusion and self.use_channel_view:
            h_region, region_attention = self.region_aggregation(h_hemi)
        else:
            h_region = x.new_zeros((n_batch, self.num_regions, d_model))
            region_attention = x.new_zeros((n_batch, self.num_regions))

        if self.use_time_view:
            z_time, time_attention = self.time_pool(h_time)
        else:
            z_time = x.new_zeros((n_batch, d_model))
            time_attention = x.new_zeros((n_batch, self.window_size))
        if self.use_freq_view:
            z_freq, freq_attention = self.freq_pool(h_freq)
        else:
            z_freq = x.new_zeros((n_batch, d_model))
            freq_attention = x.new_zeros((n_batch, self.num_bands))
        if self.use_channel_view:
            z_channel, channel_attention = self.channel_pool(h_channel)
        else:
            z_channel = x.new_zeros((n_batch, d_model))
            channel_attention = x.new_zeros((n_batch, self.num_channels))
        z_hemi = self._weighted_pool(h_hemi, pair_attention)
        z_region = self._weighted_pool(h_region, region_attention)

        z_views = torch.stack([z_time, z_freq, z_channel, z_hemi, z_region], dim=1)
        view_logits = self.view_gate(z_views).squeeze(-1)
        view_mask = torch.tensor(
            [
                self.use_time_view,
                self.use_freq_view,
                self.use_channel_view,
                self.use_hemi_fusion and self.use_channel_view,
                self.use_region_aggregation and self.use_hemi_fusion and self.use_channel_view,
            ],
            dtype=torch.bool,
            device=x.device,
        )
        view_logits = view_logits.masked_fill(~view_mask.view(1, -1), -1e4)
        view_weights = torch.softmax(view_logits, dim=1)
        z_fused = torch.einsum('bv,bvd->bd', view_weights, z_views)

        logits_direct = self.direct_branch(z_fused) if self.use_direct else z_fused.new_zeros((n_batch, self.num_classes))
        proto = self.proto_branch(z_fused) if self.use_proto else self._zero_proto(z_fused)
        edge = self._forward_edge(h_time, h_freq, h_hemi, h_region, proto)
        concept = self.concept_branch(z_freq, z_hemi, z_region) if self.use_concept else self._zero_concept(z_fused)
        domain_logits = (
            self.subject_discriminator(self.grl(z_fused))
            if self.use_domain_adversarial
            else z_fused.new_zeros((n_batch, self.num_domains))
        )

        logits_proto = proto['logits'] if self.use_proto else z_fused.new_zeros((n_batch, self.num_classes))
        logits_edge = edge['logits'] if self.use_edge else z_fused.new_zeros((n_batch, self.num_classes))
        logits_concept = concept['logits'] if self.use_concept else z_fused.new_zeros((n_batch, self.num_classes))
        logits_final = self._combine_logits(logits_direct, logits_proto, logits_edge, logits_concept)

        outputs = {
            'logits': logits_final,
            'logits_final': logits_final,
            'logits_direct': logits_direct,
            'logits_proto': logits_proto,
            'logits_edge': logits_edge,
            'logits_concept': logits_concept,
            'domain_logits': domain_logits,
            'z_fused': z_fused,
            'proto_embedding': proto['z_hyperbolic'],
            'proto_distance': proto['distance'],
            'proto_tangent': proto['v'],
            'prototypes': proto['prototypes'],
            'concept_scores': concept['concept_scores'],
            'view_weights': view_weights,
            'time_attention': time_attention,
            'freq_attention': freq_attention,
            'channel_attention': channel_attention,
            'pair_attention': pair_attention,
            'region_attention': region_attention,
            'edge_attention': edge['edge_attention'],
            'edge_weights': edge['edge_weights'],
            'edge_evidence': edge['edge_evidence'],
            'edge_neutral_evidence': edge['neutral_evidence'],
            'edge_time_attention': edge['edge_time_attention'],
            'edge_freq_attention': edge['edge_freq_attention'],
            'edge_pair_attention': edge['edge_pair_attention'],
            'edge_region_attention': edge['edge_region_attention'],
            'h_time': h_time,
            'h_freq': h_freq,
            'h_channel': h_channel,
            'h_hemi': h_hemi,
            'h_region': h_region,
            'h_tok': torch.cat([h_time, h_freq, h_hemi, h_region], dim=1),
            'h_struct': h_time.unsqueeze(2).unsqueeze(3),
        }
        for key in (
            'edge_relation',
            'edge_query',
            'edge_distance',
            'edge_scores',
            'edge_logits',
            'edge_code_logits',
            'edge_endpoint_logits',
            'edge_graph_logits',
            'raw_edge_evidence_score',
            'edge_evidence_score',
            'edge_relation_bias',
            'edge_log_probs',
            'path_log_probs',
            'path_prob',
            'path_matrix',
            'node_distance_gain',
            'class_path_energy',
            'stop_score',
            'root_logits',
            'root_prob',
            'child_logits',
            'child_prob',
        ):
            if key in edge:
                outputs[key] = edge[key]
        if 'prototype_anchor' in proto:
            outputs['prototype_anchor'] = proto['prototype_anchor']
        return outputs
