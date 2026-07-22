import unittest

import torch
import torch.nn.functional as F

from graphs.affective_graph import get_affective_graph
from losses.eagle_loss import compute_eagle_loss
from models import build_model
from utils.config import load_config


class NeutralCenteredPathTests(unittest.TestCase):
    def test_seediv_graph_uses_neutral_root_and_full_happy_path(self):
        graph = get_affective_graph('SEED-IV', 'seediv_neutral_centered_path_graph')

        self.assertEqual(graph['root_node'], 'neutral')
        self.assertEqual(graph['class_node_indices'][0], 0)
        self.assertEqual(graph['fixed_node_indices'], [0])
        self.assertEqual(graph['path_matrix'][0], [0, 0, 0, 0, 0])
        self.assertEqual(graph['path_matrix'][1], [1, 0, 1, 0, 0])
        self.assertEqual(graph['root_class_groups'], [[0], [1], [2, 3]])

    def test_leaf_probability_decomposes_into_root_and_child(self):
        cfg = load_config('configs/seediv_neutral_centered_path_eagle_probe.yaml')
        graph = get_affective_graph('SEED-IV', cfg['graph']['name'])
        outputs = build_model(cfg, graph).eval()(torch.randn(4, 10, 62, 5))
        labels = torch.tensor([0, 1, 2, 3])

        leaf_nll = -outputs['path_log_probs'][torch.arange(4), labels]
        root_targets = torch.tensor(graph['root_target_by_class']).index_select(0, labels)
        expected = -F.log_softmax(outputs['root_logits'], dim=1)[torch.arange(4), root_targets]
        child_log_prob = F.log_softmax(outputs['child_logits'], dim=1)
        expected[2:] += -child_log_prob[torch.arange(2, 4), torch.tensor([0, 1])]

        self.assertTrue(torch.allclose(leaf_nll, expected, atol=1e-5))
        self.assertTrue(torch.allclose(outputs['path_prob'].sum(dim=1), torch.ones(4), atol=1e-6))

    def test_neutral_prototype_is_fixed_at_origin(self):
        cfg = load_config('configs/seediv_neutral_centered_path_eagle_probe.yaml')
        graph = get_affective_graph('SEED-IV', cfg['graph']['name'])
        model = build_model(cfg, graph)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        F.cross_entropy(outputs['logits_edge'], labels).backward()

        self.assertEqual(model.proto_branch.prototype_offset.grad[0].abs().sum().item(), 0.0)
        optimizer.step()
        self.assertTrue(torch.equal(model.proto_branch.prototypes()[0], torch.zeros(32)))

    def test_hierarchical_losses_and_seed_shapes(self):
        seediv_cfg = load_config('configs/seediv_neutral_centered_path_eagle_probe.yaml')
        seediv_graph = get_affective_graph('SEED-IV', seediv_cfg['graph']['name'])
        seediv_model = build_model(seediv_cfg, seediv_graph)
        labels = torch.tensor([0, 1, 2, 3, 2, 3])
        outputs = seediv_model(torch.randn(6, 10, 62, 5))
        total, parts = compute_eagle_loss(
            outputs,
            labels,
            seediv_graph,
            seediv_cfg,
            subject_ids=torch.arange(6),
            epoch=1,
        )

        self.assertGreater(parts['hier_root'].item(), 0.0)
        self.assertGreater(parts['hier_child'].item(), 0.0)
        self.assertTrue(torch.isfinite(total).item())

        seed_cfg = load_config('configs/seed_neutral_centered_path_eagle_probe.yaml')
        seed_graph = get_affective_graph('SEED', seed_cfg['graph']['name'])
        seed_outputs = build_model(seed_cfg, seed_graph)(torch.randn(2, 30, 62, 5))
        self.assertEqual(seed_outputs['root_logits'].shape, (2, 3))
        self.assertEqual(seed_outputs['child_logits'].shape, (2, 0))


if __name__ == '__main__':
    unittest.main()
