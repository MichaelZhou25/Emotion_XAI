import unittest

import torch

from graphs.affective_graph import get_affective_graph
from losses.eagle_loss import compute_eagle_loss
from models import build_model
from utils.config import load_config


class SeedIVNoConceptTests(unittest.TestCase):
    def test_multiview_hierarchy_shapes_and_loss(self):
        cfg = load_config('configs/smoke_seediv_no_concept.yaml')
        graph = get_affective_graph('SEED-IV')
        model = build_model(cfg, graph)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model(torch.randn(4, 10, 62, 5))
        loss, _ = compute_eagle_loss(
            outputs,
            labels,
            graph,
            cfg,
            subject_ids=torch.arange(4),
            epoch=1,
        )

        self.assertEqual(outputs['logits'].shape, (4, 4))
        self.assertEqual(outputs['proto_distance'].shape, (4, 6))
        self.assertEqual(outputs['prototypes'].shape, (6, 32))
        self.assertEqual(outputs['edge_weights'].shape, (4, 5))
        self.assertEqual(outputs['edge_attention'].shape, (4, 5, 47))
        self.assertTrue(torch.count_nonzero(outputs['concept_scores']).item() == 0)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()
