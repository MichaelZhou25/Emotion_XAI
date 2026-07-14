import tempfile
import unittest
from pathlib import Path

import torch

from losses.eagle_loss import _scheduled_weight
from losses.hyperbolic_contrastive_loss import hyperbolic_subject_centroid_loss
from models.hyperbolic_prototype import expmap0
from models.hemi_mv_eagle_net import GradientReversal
from trainer.engine import build_lr_scheduler
from utils.config import load_config
from utils.ema import ModelEMA
from utils.schedules import scheduled_value


class ScheduleTests(unittest.TestCase):
    def test_linear_schedule(self):
        self.assertEqual(scheduled_value(0.2, 5, 'linear', 5, 25), 0.0)
        self.assertAlmostEqual(scheduled_value(0.2, 6, 'linear', 5, 25), 0.008)
        self.assertAlmostEqual(scheduled_value(0.2, 30, 'linear', 5, 25), 0.2)

    def test_domain_does_not_inherit_hp_schedule(self):
        cfg = {'lambda_domain': 0.05, 'hp_schedule': 'linear', 'hp_warmup_epochs': 10}
        self.assertEqual(_scheduled_weight(cfg, 'domain', 1), 0.05)

    def test_cosine_scheduler_warmup_and_floor(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        cfg = {
            'train': {
                'epochs': 10,
                'lr': 1e-3,
                'lr_scheduler': 'cosine',
                'lr_warmup_epochs': 2,
                'min_lr': 1e-4,
            }
        }
        scheduler = build_lr_scheduler(optimizer, cfg)
        values = [optimizer.param_groups[0]['lr']]
        for _ in range(9):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(values[0], 5e-4)
        self.assertAlmostEqual(values[1], 1e-3)
        self.assertAlmostEqual(values[-1], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(values[1:], values[2:])))


class TrainingStateTests(unittest.TestCase):
    def test_gradient_reversal_runtime_lambda(self):
        layer = GradientReversal(0.2)
        layer.set_lambda(0.4)
        x = torch.ones(1, requires_grad=True)
        layer(x).sum().backward()
        self.assertAlmostEqual(x.grad.item(), -0.4)

    def test_ema_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.5, warmup_updates=0)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        self.assertTrue(torch.allclose(ema.module.weight, torch.full_like(model.weight, 0.5)))

    def test_subject_centroid_loss_is_finite_and_differentiable(self):
        tangent = (torch.randn(8, 4) * 0.05).requires_grad_()
        prototype_tangent = (torch.randn(3, 4) * 0.05).requires_grad_()
        outputs = {
            'proto_tangent': tangent,
            'proto_embedding': expmap0(tangent),
            'prototypes': expmap0(prototype_tangent),
        }
        graph = {
            'num_classes': 3,
            'num_nodes': 3,
            'nodes': ['neutral', 'positive', 'negative'],
            'edges': [('neutral', 'positive'), ('neutral', 'negative')],
            'node_depth': [0, 1, 1],
            'concept_matrix': [[0.0], [1.0], [-1.0]],
            'class_node_indices': [0, 1, 2],
        }
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 2, 2])
        subject_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = hyperbolic_subject_centroid_loss(outputs, labels, subject_ids, graph)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(tangent.grad)


class ConfigTests(unittest.TestCase):
    def test_relative_extends_and_deep_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'base.yaml').write_text(
                'train:\n  lr: 0.001\n  epochs: 10\nmodel:\n  d_model: 64\n',
                encoding='utf-8',
            )
            (root / 'child.yaml').write_text(
                'extends: base.yaml\ntrain:\n  lr: 0.0005\n',
                encoding='utf-8',
            )
            cfg = load_config(root / 'child.yaml')
        self.assertEqual(cfg['train']['lr'], 0.0005)
        self.assertEqual(cfg['train']['epochs'], 10)
        self.assertEqual(cfg['model']['d_model'], 64)


if __name__ == '__main__':
    unittest.main()
