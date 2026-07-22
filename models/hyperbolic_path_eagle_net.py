import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hemi_mv_eagle_net import HemiMVEAGLENet
from models.hyperbolic_prototype import expmap0, poincare_distance, project_ball


def mobius_add(x, y, eps=1e-8):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    denominator = (1.0 + 2.0 * xy + x2 * y2).clamp_min(eps)
    return project_ball(numerator / denominator)


def logmap0(x, eps=1e-8):
    x = project_ball(x)
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.atanh(norm.clamp_max(1.0 - 1e-5)) / norm
    return scale * x


class GraphAnchoredHyperbolicPrototypeBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        model_cfg = cfg['model']
        d_model = model_cfg['d_model']
        d_proto = model_cfg.get('d_proto', 32)
        self.temperature = model_cfg.get('proto_temperature', 0.5)
        self.detach_encoder_input = bool(model_cfg.get('detach_path_encoder', False))
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proto),
        )
        self.register_buffer(
            'class_node_indices',
            torch.tensor(graph['class_node_indices'], dtype=torch.long),
        )

        coords = torch.tensor(graph['semantic_coords'], dtype=torch.float32)
        if coords.shape[1] > d_proto:
            coords = coords[:, :d_proto]
        elif coords.shape[1] < d_proto:
            coords = F.pad(coords, (0, d_proto - coords.shape[1]))
        depth = torch.tensor(graph['node_depth'], dtype=torch.float32).unsqueeze(-1)
        root_radius = float(model_cfg.get('prototype_root_radius', 0.0))
        depth_step = float(model_cfg.get('prototype_depth_step', 0.22))
        radius = root_radius + depth_step * depth
        anchor_tangent = F.normalize(coords, dim=-1, eps=1e-8) * radius
        self.register_buffer('anchor_tangent', anchor_tangent)
        offset_mask = torch.ones_like(anchor_tangent)
        for index in graph.get('fixed_node_indices', []):
            offset_mask[int(index)] = 0.0
        self.register_buffer('prototype_offset_mask', offset_mask, persistent=False)
        self.prototype_offset = nn.Parameter(torch.zeros_like(anchor_tangent))

    def prototype_tangent(self):
        return self.anchor_tangent + self.prototype_offset * self.prototype_offset_mask

    def prototypes(self):
        return expmap0(self.prototype_tangent())

    def forward(self, z_fused):
        encoder_input = z_fused.detach() if self.detach_encoder_input else z_fused
        tangent = self.project(encoder_input)
        z_hyperbolic = expmap0(tangent)
        prototypes = self.prototypes()
        distance = poincare_distance(z_hyperbolic, prototypes)
        logits_all = -distance.pow(2) / self.temperature
        logits = logits_all.index_select(1, self.class_node_indices)
        return {
            'logits': logits,
            'distance': distance,
            'z_hyperbolic': z_hyperbolic,
            'v': tangent,
            'prototypes': prototypes,
            'prototype_anchor': expmap0(self.anchor_tangent),
        }


class PrototypeGuidedPathBranch(nn.Module):
    def __init__(self, cfg, graph):
        super().__init__()
        model_cfg = cfg['model']
        d_model = model_cfg['d_model']
        d_proto = model_cfg.get('d_proto', 32)
        path_dropout = float(model_cfg.get('path_dropout', model_cfg.get('dropout', 0.0)))
        self.num_classes = graph['num_classes']
        self.num_edges = graph['num_edges']
        self.class_names = list(graph['classes'])
        self.path_scoring = graph.get('path_scoring', 'conditional')
        self.detach_encoder_input = bool(model_cfg.get('detach_path_encoder', False))
        self.bound_evidence_score = bool(model_cfg.get('path_bound_evidence_score', False))
        self.evidence_score_scale = float(model_cfg.get('path_evidence_score_scale', 1.0))
        self.use_relation_bias = bool(model_cfg.get('path_use_relation_bias', True))

        node_to_index = {node: index for index, node in enumerate(graph['nodes'])}
        root_name = graph.get('root_node')
        if root_name not in node_to_index:
            raise ValueError('Prototype path graph must define a valid root_node')
        root_index = node_to_index[root_name]

        parent_indices = []
        child_indices = []
        outgoing_edges = {}
        child_to_edge = {}
        for edge_index, (parent, child) in enumerate(graph['edges']):
            parent_index = node_to_index[parent]
            child_index = node_to_index[child]
            if child_index in child_to_edge:
                raise ValueError(f'Prototype path graph is not a tree: {child} has multiple parents')
            parent_indices.append(parent_index)
            child_indices.append(child_index)
            child_to_edge[child_index] = edge_index
            outgoing_edges.setdefault(parent_index, []).append(edge_index)

        path_matrix = torch.zeros(self.num_classes, self.num_edges, dtype=torch.float32)
        for class_index, node_index in enumerate(graph['class_node_indices']):
            current = int(node_index)
            visited = set()
            while current != root_index:
                if current in visited or current not in child_to_edge:
                    raise ValueError(f'Class node {graph["nodes"][node_index]} has no valid root path')
                visited.add(current)
                edge_index = child_to_edge[current]
                path_matrix[class_index, edge_index] = 1.0
                current = parent_indices[edge_index]

        configured_path = torch.tensor(graph['path_matrix'], dtype=torch.float32)
        if configured_path.shape != path_matrix.shape or not torch.equal(configured_path, path_matrix):
            raise ValueError('graph.path_matrix does not match paths derived from graph.edges')

        self.register_buffer('parent_indices', torch.tensor(parent_indices, dtype=torch.long))
        self.register_buffer('child_indices', torch.tensor(child_indices, dtype=torch.long))
        self.register_buffer('path_matrix', path_matrix)
        depth = torch.tensor(graph['node_depth'], dtype=torch.float32)
        depth_delta = depth[self.child_indices] - depth[self.parent_indices]
        self.register_buffer('edge_depth_delta', depth_delta)
        self.outgoing_edge_groups = [tuple(indices) for indices in outgoing_edges.values()]
        self.root_class_groups = [tuple(int(index) for index in group) for group in graph.get('root_class_groups', [])]
        child_class_indices = graph.get('child_class_indices', [])
        self.register_buffer(
            'child_class_indices',
            torch.tensor(child_class_indices, dtype=torch.long),
            persistent=False,
        )

        if self.path_scoring == 'energy':
            if sorted(index for group in self.root_class_groups for index in group) != list(range(self.num_classes)):
                raise ValueError('Energy path graph root_class_groups must partition every class exactly once')
            self.stop_class_index = int(graph['stop_class_index'])
            self.stop_node_index = int(graph['class_node_indices'][self.stop_class_index])
            if self.stop_node_index != root_index:
                raise ValueError('Neutral stop class must use the prototype root node')
            self.stop_scale_raw = nn.Parameter(torch.tensor(0.0))
            self.stop_bias = nn.Parameter(torch.tensor(0.0))
        elif self.path_scoring == 'conditional':
            self.stop_class_index = None
            self.stop_node_index = None
            self.register_parameter('stop_scale_raw', None)
            self.register_parameter('stop_bias', None)
        else:
            raise ValueError(f'Unknown path_scoring mode: {self.path_scoring}')

        relation_dim = d_proto + 2
        self.relation_to_query = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, d_model),
            nn.GELU(),
            nn.Dropout(path_dropout),
            nn.LayerNorm(d_model),
        )
        self.evidence_to_relation = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_proto),
        )
        self.evidence_score = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(path_dropout),
            nn.Linear(d_model, 1),
        )
        self.relation_bias = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, 1),
        )
        self.prototype_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.relation_scale_raw = nn.Parameter(torch.tensor(0.0))

    def _edge_relations(self, prototypes):
        parent = prototypes.index_select(0, self.parent_indices)
        child = prototypes.index_select(0, self.child_indices)
        displacement = mobius_add(-parent, child)
        relation = logmap0(displacement)
        edge_distance = torch.diagonal(poincare_distance(parent, child), 0)
        relation_features = torch.cat(
            [relation, edge_distance.unsqueeze(-1), self.edge_depth_delta.unsqueeze(-1)],
            dim=-1,
        )
        return relation, relation_features, edge_distance

    def _conditional_log_prob(self, edge_scores):
        edge_log_prob = torch.zeros_like(edge_scores)
        for group in self.outgoing_edge_groups:
            indices = torch.tensor(group, dtype=torch.long, device=edge_scores.device)
            values = F.log_softmax(edge_scores.index_select(1, indices), dim=1)
            edge_log_prob = edge_log_prob.scatter(1, indices.view(1, -1).expand_as(values), values)
        return edge_log_prob

    def _energy_path_outputs(self, edge_scores, node_distance):
        class_path_energy = torch.matmul(edge_scores, self.path_matrix.t())
        stop_scale = F.softplus(self.stop_scale_raw)
        stop_score = -stop_scale * node_distance[:, self.stop_node_index].pow(2) + self.stop_bias
        class_path_energy = class_path_energy.clone()
        class_path_energy[:, self.stop_class_index] += stop_score
        path_log_prob = F.log_softmax(class_path_energy, dim=1)

        root_logits = torch.stack(
            [torch.logsumexp(class_path_energy[:, group], dim=1) for group in self.root_class_groups],
            dim=1,
        )
        if self.child_class_indices.numel() > 0:
            child_logits = class_path_energy.index_select(1, self.child_class_indices)
        else:
            child_logits = class_path_energy.new_zeros((class_path_energy.shape[0], 0))
        return {
            'class_path_energy': class_path_energy,
            'path_log_prob': path_log_prob,
            'stop_score': stop_score,
            'root_logits': root_logits,
            'root_prob': F.softmax(root_logits, dim=1),
            'child_logits': child_logits,
            'child_prob': F.softmax(child_logits, dim=1) if child_logits.shape[1] else child_logits,
        }

    def forward(self, h_time, h_freq, h_hemi, h_region, proto):
        if self.detach_encoder_input:
            h_time = h_time.detach()
            h_freq = h_freq.detach()
            h_hemi = h_hemi.detach()
            h_region = h_region.detach()
        tokens = torch.cat([h_time, h_freq, h_hemi, h_region], dim=1)
        n_time = h_time.shape[1]
        n_freq = h_freq.shape[1]
        n_pair = h_hemi.shape[1]
        n_region = h_region.shape[1]
        n_batch, _, d_model = tokens.shape

        relation, relation_features, edge_distance = self._edge_relations(proto['prototypes'])
        query = self.relation_to_query(relation_features)
        attention_score = torch.matmul(query.unsqueeze(0), tokens.transpose(1, 2)) / math.sqrt(d_model)
        attention = torch.softmax(attention_score, dim=-1)
        evidence = torch.matmul(attention, tokens)

        query_batch = query.unsqueeze(0).expand(n_batch, -1, -1)
        evidence_features = torch.cat(
            [evidence, query_batch, evidence * query_batch, torch.abs(evidence - query_batch)],
            dim=-1,
        )
        raw_evidence_score = self.evidence_score(evidence_features).squeeze(-1)
        learned_evidence_score = raw_evidence_score
        if self.bound_evidence_score:
            learned_evidence_score = self.evidence_score_scale * torch.tanh(raw_evidence_score)
        evidence_relation = self.evidence_to_relation(evidence)
        relation_compatibility = F.cosine_similarity(
            evidence_relation,
            relation.unsqueeze(0),
            dim=-1,
            eps=1e-8,
        )

        node_distance = proto['distance']
        parent_distance = node_distance.index_select(1, self.parent_indices)
        child_distance = node_distance.index_select(1, self.child_indices)
        distance_gain = parent_distance - child_distance
        prototype_scale = F.softplus(self.prototype_scale_raw)
        relation_scale = F.softplus(self.relation_scale_raw)
        relation_bias = self.relation_bias(relation_features).squeeze(-1).unsqueeze(0)
        if not self.use_relation_bias:
            relation_bias = torch.zeros_like(relation_bias)
        edge_scores = (
            prototype_scale * distance_gain
            + relation_scale * relation_compatibility
            + learned_evidence_score
            + relation_bias
        )

        energy_outputs = None
        if self.path_scoring == 'energy':
            energy_outputs = self._energy_path_outputs(edge_scores, node_distance)
            path_log_prob = energy_outputs['path_log_prob']
            edge_log_prob = F.logsigmoid(edge_scores)
            edge_weights = torch.sigmoid(edge_scores)
        else:
            edge_log_prob = self._conditional_log_prob(edge_scores)
            class_path_score = torch.matmul(edge_log_prob, self.path_matrix.t())
            path_log_prob = F.log_softmax(class_path_score, dim=1)
            edge_weights = edge_log_prob.exp()
        path_prob = path_log_prob.exp()

        neutral_evidence = tokens.new_zeros((n_batch,))
        if 'neutral' in self.class_names:
            neutral_evidence = path_prob[:, self.class_names.index('neutral')]

        start = 0
        edge_time_attention = attention[:, :, start:start + n_time]
        start += n_time
        edge_freq_attention = attention[:, :, start:start + n_freq]
        start += n_freq
        edge_pair_attention = attention[:, :, start:start + n_pair]
        start += n_pair
        edge_region_attention = attention[:, :, start:start + n_region]

        outputs = {
            'logits': path_log_prob,
            'edge_weights': edge_weights,
            'edge_attention': attention,
            'edge_evidence': evidence,
            'neutral_evidence': neutral_evidence,
            'edge_time_attention': edge_time_attention,
            'edge_freq_attention': edge_freq_attention,
            'edge_pair_attention': edge_pair_attention,
            'edge_region_attention': edge_region_attention,
            'edge_relation': relation,
            'edge_query': query,
            'edge_distance': edge_distance,
            'edge_scores': edge_scores,
            'raw_edge_evidence_score': raw_evidence_score,
            'edge_evidence_score': learned_evidence_score,
            'edge_relation_bias': relation_bias,
            'edge_log_probs': edge_log_prob,
            'path_log_probs': path_log_prob,
            'path_prob': path_prob,
            'path_matrix': self.path_matrix,
            'node_distance_gain': distance_gain,
        }
        if energy_outputs is not None:
            outputs.update(energy_outputs)
        return outputs


class HyperbolicPathEAGLENet(HemiMVEAGLENet):
    def __init__(self, cfg, graph):
        super().__init__(cfg, graph)
        if 'root_node' not in graph:
            raise ValueError('HyperbolicPathEAGLENet requires a graph with root_node')
        self.use_proto = True
        self.use_edge = True
        self.proto_branch = GraphAnchoredHyperbolicPrototypeBranch(cfg, graph)
        self.edge_branch = PrototypeGuidedPathBranch(cfg, graph)
        weights = cfg['model'].get('branch_weights', {})
        self.w_path = float(weights.get('path', weights.get('edge', 1.0)))
        self.w_direct = float(weights.get('direct', 0.2))
        self.w_proto = float(weights.get('proto', 0.0))

    def _forward_edge(self, h_time, h_freq, h_hemi, h_region, proto):
        return self.edge_branch(h_time, h_freq, h_hemi, h_region, proto)

    def _combine_logits(self, logits_direct, logits_proto, logits_edge, logits_concept):
        return self.w_path * logits_edge + self.w_direct * logits_direct + self.w_proto * logits_proto


class NeutralCenteredHyperbolicPathEAGLENet(HyperbolicPathEAGLENet):
    def __init__(self, cfg, graph):
        if graph.get('root_node') != 'neutral' or graph.get('path_scoring') != 'energy':
            raise ValueError('NeutralCenteredHyperbolicPathEAGLENet requires a neutral-root energy graph')
        super().__init__(cfg, graph)
