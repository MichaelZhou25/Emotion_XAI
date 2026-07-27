import unittest
from copy import deepcopy

import torch

from graphs.affective_graph import get_affective_graph
from losses.eagle_loss import compute_eagle_loss
from losses.edge_loss import edge_bce_loss
from losses.edge_only_loss import (
    root_decision_loss,
    sibling_edge_contrastive_loss,
    tree_validity_loss,
)
from losses.hyperbolic_contrastive_loss import hyperbolic_subject_centroid_loss
from models import build_model
from models.edge_code import bernoulli_edge_code_logits
from utils.config import load_config


class HyperbolicEdgeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config('configs/seediv_hyperbolic_edge_only_probe.yaml')
        self.graph = get_affective_graph('SEED-IV', self.cfg['graph']['name'])

    def test_fixed_edge_code_decoder_recovers_all_classes(self):
        codebook = torch.tensor(self.graph['path_matrix'], dtype=torch.float32)
        edge_logits = (2.0 * codebook - 1.0) * 12.0
        class_logits = bernoulli_edge_code_logits(edge_logits, codebook)

        self.assertTrue(torch.equal(class_logits.argmax(dim=1), torch.arange(4)))

    def test_tree_validity_penalizes_invalid_edge_patterns(self):
        codebook = torch.tensor(self.graph['path_matrix'], dtype=torch.float32)
        valid_logits = (2.0 * codebook - 1.0) * 20.0
        invalid_logits = torch.tensor([[-20.0, -20.0, 20.0, 20.0, 20.0]])

        valid_loss = tree_validity_loss(valid_logits, self.graph)
        invalid_loss = tree_validity_loss(invalid_logits, self.graph)

        self.assertLess(valid_loss.item(), 1e-6)
        self.assertGreater(invalid_loss.item(), 0.5)

    def test_root_decision_includes_neutral_stop_state(self):
        labels = torch.tensor([0, 1, 2, 3])
        edge_logits = torch.tensor([
            [-20.0, -20.0, -20.0, -20.0, -20.0],
            [20.0, -20.0, 20.0, -20.0, -20.0],
            [-20.0, 20.0, -20.0, 20.0, -20.0],
            [-20.0, 20.0, -20.0, -20.0, 20.0],
        ])
        correct = root_decision_loss(edge_logits, labels, self.graph)
        wrong = root_decision_loss(edge_logits[[1, 0, 3, 2]], labels, self.graph)

        self.assertLess(correct.item(), 1e-5)
        self.assertGreater(wrong.item(), 5.0)

    def test_child_decision_only_supervises_sad_and_fear(self):
        labels = torch.tensor([0, 1, 2, 3])
        correct_logits = torch.tensor([
            [-20.0, -20.0, -20.0, -20.0, -20.0],
            [20.0, -20.0, 20.0, -20.0, -20.0],
            [-20.0, 20.0, -20.0, 20.0, -20.0],
            [-20.0, 20.0, -20.0, -20.0, 20.0],
        ])
        wrong_logits = correct_logits.clone()
        wrong_logits[2:, 3:5] = wrong_logits[2:, [4, 3]]

        correct = sibling_edge_contrastive_loss(
            correct_logits,
            labels,
            self.graph,
            min_parent_depth=1,
        )
        wrong = sibling_edge_contrastive_loss(
            wrong_logits,
            labels,
            self.graph,
            min_parent_depth=1,
        )

        self.assertLess(correct.item(), 1e-5)
        self.assertGreater(wrong.item(), 20.0)

    def test_edge_loss_weights_validate_edge_count(self):
        labels = torch.tensor([0, 1, 2, 3])
        edge_logits = torch.zeros(4, 5)
        with self.assertRaisesRegex(ValueError, 'must contain 5 values'):
            edge_bce_loss(
                torch.sigmoid(edge_logits),
                labels,
                self.graph,
                edge_scores=edge_logits,
                loss_weights=[1.0, 1.0],
            )

    def test_model_outputs_only_edge_decoded_class_logits(self):
        model = build_model(self.cfg, self.graph).eval()
        outputs = model(torch.randn(3, 10, 62, 5))

        self.assertEqual(outputs['logits'].shape, (3, 4))
        self.assertEqual(outputs['edge_logits'].shape, (3, 5))
        self.assertTrue(torch.allclose(outputs['logits'], outputs['edge_code_logits']))
        self.assertTrue(torch.allclose(outputs['logits_final'], outputs['logits_edge']))
        self.assertNotIn('stop_score', outputs)
        parameter_names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any(name.startswith('direct_branch.') for name in parameter_names))
        self.assertFalse(any(name.startswith('concept_branch.') for name in parameter_names))

    def test_legacy_view_identity_flag_still_enables_both_embeddings(self):
        cfg = deepcopy(self.cfg)
        cfg['model']['use_view_identity_embeddings'] = True
        model = build_model(cfg, self.graph)

        self.assertIsNotNone(model.freq_identity)
        self.assertIsNotNone(model.channel_identity)

    def test_loss_has_exactly_four_active_terms_and_backpropagates(self):
        model = build_model(self.cfg, self.graph)
        labels = torch.tensor([0, 1, 2, 3])
        subject_ids = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        total, parts = compute_eagle_loss(
            outputs,
            labels,
            self.graph,
            self.cfg,
            subject_ids=subject_ids,
            epoch=1,
        )

        self.assertEqual(set(parts), {'edge', 'tree', 'hpcl', 'domain', 'total'})
        expected = (
            parts['edge']
            + 0.05 * parts['tree']
            + 0.02 * parts['hpcl']
            + 0.05 * parts['domain']
        )
        self.assertTrue(torch.allclose(total, expected))
        total.backward()
        self.assertTrue(torch.isfinite(model.proto_branch.prototype_offset.grad).all())
        self.assertGreater(model.proto_branch.prototype_offset.grad.abs().sum().item(), 0.0)
        evidence_grad = model.edge_branch.evidence_score[-1].weight.grad
        self.assertIsNotNone(evidence_grad)
        self.assertGreater(evidence_grad.abs().sum().item(), 0.0)

    def test_subject_centroid_alignment_is_internal_to_hpcl(self):
        cfg = deepcopy(self.cfg)
        cfg['loss']['hp_subject_centroid_weight'] = 0.5
        cfg['loss']['hp_subject_centroid_schedule'] = 'constant'
        model = build_model(cfg, self.graph)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        subject_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        outputs = model(torch.randn(8, 10, 62, 5))
        _, base_parts = compute_eagle_loss(
            outputs,
            labels,
            self.graph,
            self.cfg,
            subject_ids=subject_ids,
            epoch=10,
        )
        _, parts = compute_eagle_loss(
            outputs,
            labels,
            self.graph,
            cfg,
            subject_ids=subject_ids,
            epoch=10,
        )
        centroid = hyperbolic_subject_centroid_loss(
            outputs,
            labels,
            subject_ids,
            self.graph,
            base_margin=0.2,
            max_margin=0.6,
            neutral_boost=0.1,
            branch_boost=0.05,
            same_branch_scale=0.5,
            pull_weight=1.0,
            margin_weight=0.5,
            compactness_weight=0.1,
            pair_weight_base=1.0,
        )

        self.assertEqual(set(parts), {'edge', 'tree', 'hpcl', 'domain', 'total'})
        self.assertTrue(torch.allclose(parts['hpcl'], base_parts['hpcl'] + 0.5 * centroid))

    def test_code_ce_objective_directly_supervises_fixed_decoder(self):
        cfg = load_config('configs/seediv_hyperbolic_edge_only_v2_fixed_probe.yaml')
        model = build_model(cfg, self.graph)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        total, parts = compute_eagle_loss(outputs, labels, self.graph, cfg)
        expected_edge = torch.nn.functional.cross_entropy(
            outputs['edge_code_logits'],
            labels,
        )

        self.assertTrue(torch.allclose(parts['edge'], expected_edge))
        self.assertTrue(torch.isfinite(total))

    def test_hybrid_edge_objective_combines_local_and_global_code_loss(self):
        cfg = load_config('configs/seediv_hyperbolic_edge_only_v3_fixed_probe.yaml')
        model = build_model(cfg, self.graph)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        _, bce_parts = compute_eagle_loss(outputs, labels, self.graph, self.cfg)
        _, hybrid_parts = compute_eagle_loss(outputs, labels, self.graph, cfg)
        code_ce = torch.nn.functional.cross_entropy(outputs['edge_code_logits'], labels)

        expected = bce_parts['edge'] + 0.2 * code_ce
        self.assertTrue(torch.allclose(hybrid_parts['edge'], expected))

    def test_sibling_contrast_only_compares_edges_on_the_target_path(self):
        labels = torch.tensor([0, 1, 2, 3])
        correct_logits = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, -8.0, 8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0, 8.0, -8.0],
            [-8.0, 8.0, -8.0, -8.0, 8.0],
        ])
        wrong_logits = correct_logits.clone()
        wrong_logits[1:, [0, 1]] = wrong_logits[1:, [1, 0]]
        wrong_logits[2:, [3, 4]] = wrong_logits[2:, [4, 3]]

        correct = sibling_edge_contrastive_loss(correct_logits, labels, self.graph)
        wrong = sibling_edge_contrastive_loss(wrong_logits, labels, self.graph)

        self.assertLess(correct.item(), 1e-4)
        self.assertGreater(wrong.item(), 5.0)

    def test_leaf_sibling_contrast_ignores_root_edge_order(self):
        labels = torch.tensor([1, 2, 3])
        logits = torch.tensor([
            [-8.0, 8.0, 8.0, -8.0, -8.0],
            [8.0, -8.0, -8.0, 8.0, -8.0],
            [8.0, -8.0, -8.0, -8.0, 8.0],
        ])
        root_swapped = logits.clone()
        root_swapped[:, [0, 1]] = root_swapped[:, [1, 0]]

        original = sibling_edge_contrastive_loss(
            logits,
            labels,
            self.graph,
            min_parent_depth=1,
        )
        swapped = sibling_edge_contrastive_loss(
            root_swapped,
            labels,
            self.graph,
            min_parent_depth=1,
        )

        self.assertTrue(torch.allclose(original, swapped))

    def test_node_edge_graph_energy_has_no_direct_classifier(self):
        cfg = load_config('configs/seediv_hyperbolic_node_edge_v8_probe.yaml')
        model = build_model(cfg, self.graph)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        _, parts = compute_eagle_loss(outputs, labels, self.graph, cfg)

        expected_logits = (
            outputs['edge_code_logits']
            + 0.5 * outputs['edge_endpoint_logits']
        )
        expected_edge_loss = (
            torch.nn.functional.cross_entropy(expected_logits, labels)
            + 0.2 * edge_bce_loss(
                outputs['edge_weights'],
                labels,
                self.graph,
                edge_scores=outputs['edge_logits'],
                positive_weight=1.0,
            )
        )
        self.assertTrue(torch.allclose(outputs['logits'], expected_logits))
        self.assertTrue(torch.allclose(parts['edge'], expected_edge_loss))
        self.assertIsNone(model.direct_branch)

    def test_adaptive_endpoint_gate_starts_from_fixed_v8_fusion(self):
        base_cfg = load_config('configs/seediv_hyperbolic_node_edge_v8_probe.yaml')
        cfg = deepcopy(base_cfg)
        cfg['model']['use_adaptive_endpoint_fusion'] = True
        cfg['model']['edge_endpoint_gate_hidden'] = 8
        cfg['model']['edge_endpoint_gate_max_weight'] = 1.0
        torch.manual_seed(1234)
        baseline = build_model(base_cfg, self.graph)
        torch.manual_seed(1234)
        model = build_model(cfg, self.graph)
        baseline_state = baseline.state_dict()
        gated_state = model.state_dict()
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, gated_state[name]), name)

        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))

        expected = (
            outputs['edge_code_logits']
            + 0.5 * outputs['edge_endpoint_logits']
        )
        self.assertEqual(outputs['edge_endpoint_weights'].shape, (4, 4))
        self.assertTrue(torch.allclose(outputs['edge_endpoint_weights'], torch.full((4, 4), 0.5)))
        self.assertTrue(torch.allclose(outputs['logits'], expected))

        loss = torch.nn.functional.cross_entropy(outputs['logits'], labels)
        loss.backward()
        gate_grad = model.edge_branch.endpoint_gate[-1].weight.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(gate_grad.abs().sum().item(), 0.0)

if __name__ == '__main__':
    unittest.main()
