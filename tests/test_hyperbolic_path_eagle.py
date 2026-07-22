import unittest

import torch
import torch.nn.functional as F

from graphs.affective_graph import get_affective_graph
from losses.eagle_loss import compute_eagle_loss
from models import build_model
from utils.config import load_config


class HyperbolicPathEAGLETests(unittest.TestCase):
    def test_seed_graph_has_separate_root_and_nonempty_class_paths(self):
        graph = get_affective_graph('SEED', 'seed_hyperbolic_path_graph')
        self.assertEqual(graph['root_node'], 'emotion_root')
        self.assertNotIn(0, graph['class_node_indices'])
        self.assertTrue(torch.all(graph['path_matrix_tensor'].sum(dim=1) > 0).item())

    def test_seed_forward_probabilities_and_gradient_coupling(self):
        cfg = load_config('configs/seed_hyperbolic_path_eagle_full100.yaml')
        graph = get_affective_graph('SEED', cfg['graph']['name'])
        model = build_model(cfg, graph)
        labels = torch.tensor([0, 1, 2])
        outputs = model(torch.randn(3, 30, 62, 5))
        loss, _ = compute_eagle_loss(outputs, labels, graph, cfg, epoch=1)
        loss.backward()

        self.assertEqual(outputs['logits'].shape, (3, 3))
        self.assertEqual(outputs['prototypes'].shape, (4, 32))
        self.assertEqual(outputs['edge_relation'].shape, (3, 32))
        self.assertEqual(outputs['edge_attention'].shape, (3, 3, 67))
        self.assertTrue(torch.allclose(outputs['path_prob'].sum(dim=1), torch.ones(3), atol=1e-6))
        self.assertIsNotNone(model.proto_branch.prototype_offset.grad)
        self.assertGreater(model.proto_branch.prototype_offset.grad.abs().sum().item(), 0.0)
        self.assertFalse(any(name.endswith('edge_queries') for name, _ in model.named_parameters()))
        self.assertTrue(torch.isfinite(loss).item())

    def test_seediv_sibling_edges_are_conditionally_normalized(self):
        cfg = load_config('configs/seediv_hyperbolic_path_eagle_full100.yaml')
        graph = get_affective_graph('SEED-IV', cfg['graph']['name'])
        model = build_model(cfg, graph)
        outputs = model(torch.randn(2, 10, 62, 5))
        edge_prob = outputs['edge_log_probs'].exp()

        self.assertEqual(outputs['logits'].shape, (2, 4))
        self.assertEqual(outputs['prototypes'].shape, (6, 32))
        self.assertEqual(outputs['edge_relation'].shape, (5, 32))
        self.assertTrue(torch.allclose(edge_prob[:, :3].sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(edge_prob[:, 3:].sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(outputs['path_prob'].sum(dim=1), torch.ones(2), atol=1e-6))

    def test_path_logits_depend_on_both_eeg_and_prototypes(self):
        cfg = load_config('configs/seed_hyperbolic_path_eagle_full100.yaml')
        graph = get_affective_graph('SEED', cfg['graph']['name'])
        model = build_model(cfg, graph).eval()
        x = torch.randn(2, 30, 62, 5)
        with torch.no_grad():
            first = model(x)
            model.proto_branch.prototype_offset[1, 0] += 0.3
            second = model(x)
            third = model(x + 0.2 * torch.randn_like(x))

        self.assertFalse(torch.allclose(first['edge_relation'], second['edge_relation']))
        self.assertFalse(torch.allclose(first['logits_edge'], second['logits_edge']))
        self.assertFalse(torch.allclose(second['edge_evidence'], third['edge_evidence']))

    def test_v3_path_supervision_and_auxiliary_warmup(self):
        cfg = load_config('configs/seed_hyperbolic_path_eagle_v3_probe.yaml')
        graph = get_affective_graph('SEED', cfg['graph']['name'])
        model = build_model(cfg, graph)
        labels = torch.tensor([0, 1, 2])
        subject_ids = torch.tensor([0, 1, 2])
        outputs = model(torch.randn(3, 30, 62, 5))

        _, warmup = compute_eagle_loss(
            outputs, labels, graph, cfg, subject_ids=subject_ids, epoch=1,
        )
        total, active = compute_eagle_loss(
            outputs, labels, graph, cfg, subject_ids=subject_ids, epoch=10,
        )

        self.assertGreater(warmup['ce_path'].item(), 0.0)
        self.assertEqual(warmup['hp_proto_nce_weight'].item(), 0.0)
        self.assertEqual(warmup['domain_weight'].item(), 0.0)
        self.assertGreater(active['hp_proto_nce_weight'].item(), 0.0)
        self.assertGreater(active['domain_weight'].item(), 0.0)
        self.assertTrue(torch.isfinite(total).item())

    def test_v4_bounds_free_evidence_and_disables_relation_bias(self):
        cfg = load_config('configs/seed_hyperbolic_path_eagle_v4_probe.yaml')
        graph = get_affective_graph('SEED', cfg['graph']['name'])
        outputs = build_model(cfg, graph)(torch.randn(3, 30, 62, 5))

        self.assertLessEqual(outputs['edge_evidence_score'].abs().max().item(), 1.0)
        self.assertTrue(torch.equal(outputs['edge_relation_bias'], torch.zeros_like(outputs['edge_relation_bias'])))

    def test_v6_path_gradient_is_decoupled_from_view_encoder(self):
        cfg = load_config('configs/seed_hyperbolic_path_eagle_v6_probe.yaml')
        graph = get_affective_graph('SEED', cfg['graph']['name'])
        model = build_model(cfg, graph)
        outputs = model(torch.randn(3, 30, 62, 5))

        F.cross_entropy(outputs['logits_edge'], torch.tensor([0, 1, 2])).backward()

        encoder_grad = model.time_embed[0].weight.grad
        self.assertTrue(encoder_grad is None or encoder_grad.abs().sum().item() == 0.0)
        self.assertGreater(model.proto_branch.project[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.edge_branch.evidence_score[1].weight.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
