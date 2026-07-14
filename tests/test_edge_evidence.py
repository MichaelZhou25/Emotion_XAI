import unittest

import torch

from graphs.affective_graph import get_affective_graph
from models.hemi_mv_eagle_net import MultiViewEdgeBranch


class EdgeEvidenceTests(unittest.TestCase):
    @staticmethod
    def _run(mode):
        torch.manual_seed(7)
        cfg = {
            'model': {
                'd_model': 8,
                'edge_logit_mode': mode,
            }
        }
        branch = MultiViewEdgeBranch(cfg, get_affective_graph('SEED'))
        inputs = [
            torch.randn(3, 4, 8),
            torch.randn(3, 2, 8),
            torch.randn(3, 3, 8),
            torch.randn(3, 2, 8),
        ]
        return branch(*inputs)

    def test_path_mode_preserves_original_seed_mapping(self):
        output = self._run('path')
        weights = output['edge_weights']
        expected = torch.stack(
            [torch.zeros_like(weights[:, 0]), weights[:, 0], weights[:, 1]],
            dim=1,
        )
        self.assertTrue(torch.allclose(output['logits'], expected))

    def test_neutral_evidence_activates_only_root_class(self):
        output = self._run('neutral_evidence')
        weights = output['edge_weights']
        neutral = (1.0 - weights[:, 0]) * (1.0 - weights[:, 1])
        expected = torch.stack([neutral, weights[:, 0], weights[:, 1]], dim=1)
        self.assertTrue(torch.allclose(output['neutral_evidence'], neutral))
        self.assertTrue(torch.allclose(output['logits'], expected))

    def test_invalid_mode_is_rejected(self):
        cfg = {'model': {'d_model': 8, 'edge_logit_mode': 'invalid'}}
        with self.assertRaisesRegex(ValueError, 'edge_logit_mode'):
            MultiViewEdgeBranch(cfg, get_affective_graph('SEED'))


if __name__ == '__main__':
    unittest.main()
