import tempfile
import unittest
from pathlib import Path

import torch

from losses.eagle_loss import _scheduled_weight
from losses.domain_alignment_loss import linear_mmd_loss
from losses.hyperbolic_contrastive_loss import hyperbolic_subject_centroid_loss
from models.hyperbolic_prototype import expmap0
from models.hemi_mv_eagle_net import GradientReversal
from trainer.engine import build_lr_scheduler
from trainer.train_one_epoch import (
    _merge_target_domain_loss,
    _target_domain_weight,
    _target_feature_alignment_loss,
)
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

    def test_cosine_decay_horizon_is_independent_of_training_epochs(self):
        def lr_values(epochs, decay_epochs=None):
            parameter = torch.nn.Parameter(torch.ones(()))
            optimizer = torch.optim.AdamW([parameter], lr=1e-3)
            train_cfg = {
                'epochs': epochs,
                'lr': 1e-3,
                'lr_scheduler': 'cosine',
                'lr_warmup_epochs': 2,
                'min_lr': 1e-4,
            }
            if decay_epochs is not None:
                train_cfg['lr_decay_epochs'] = decay_epochs
            scheduler = build_lr_scheduler(optimizer, {'train': train_cfg})
            values = [optimizer.param_groups[0]['lr']]
            for _ in range(9):
                optimizer.step()
                scheduler.step()
                values.append(optimizer.param_groups[0]['lr'])
            return values

        short_run = lr_values(10)
        long_run = lr_values(100, decay_epochs=10)
        self.assertEqual(short_run, long_run)


class TrainingStateTests(unittest.TestCase):
    def test_target_domain_replaces_source_only_domain_component(self):
        source_domain = torch.tensor(2.0)
        target_domain = torch.tensor(4.0)
        total = torch.tensor(3.0)
        cfg = {
            'train': {
                'target_adaptation': {'target_domain_weight': 1.0},
            },
            'loss': {'lambda_domain': 0.05},
        }
        merged_total, parts = _merge_target_domain_loss(
            total,
            {'edge': torch.tensor(1.0), 'domain': source_domain},
            target_domain,
            cfg,
        )

        self.assertAlmostEqual(parts['domain'].item(), 3.0)
        self.assertAlmostEqual(merged_total.item(), 3.05)
        self.assertTrue(torch.equal(parts['total'], merged_total))

    def test_target_domain_weight_can_warm_up_independently(self):
        cfg = {
            'train': {
                'target_adaptation': {
                    'target_domain_weight': 1.0,
                    'schedule': 'linear',
                    'warmup_epochs': 5,
                    'ramp_epochs': 10,
                },
            },
        }
        self.assertEqual(_target_domain_weight(cfg, 5), 0.0)
        self.assertAlmostEqual(_target_domain_weight(cfg, 10), 0.5)
        self.assertEqual(_target_domain_weight(cfg, 15), 1.0)

    def test_target_feature_alignment_is_internal_to_domain_loss(self):
        cfg = {
            'train': {
                'target_adaptation': {
                    'target_domain_weight': 1.0,
                    'feature_alignment_weight': 2.0,
                },
            },
            'loss': {'lambda_domain': 0.05},
        }
        merged_total, parts = _merge_target_domain_loss(
            torch.tensor(3.0),
            {'domain': torch.tensor(2.0)},
            torch.tensor(4.0),
            cfg,
            target_feature_alignment_loss=torch.tensor(0.5),
        )

        self.assertAlmostEqual(parts['domain'].item(), 4.0)
        self.assertAlmostEqual(merged_total.item(), 3.1, places=6)
        self.assertAlmostEqual(
            parts['target_feature_alignment'].item(),
            0.5,
        )

    def test_linear_mmd_uses_feature_means(self):
        source = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        target = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
        self.assertAlmostEqual(
            linear_mmd_loss(source, target).item(),
            0.5,
        )
        cfg = {
            'train': {
                'target_adaptation': {
                    'feature_alignment': 'linear_mmd',
                },
            },
        }
        loss = _target_feature_alignment_loss(
            {'z_fused': source},
            {'z_fused': target},
            cfg,
        )
        self.assertTrue(torch.equal(loss, linear_mmd_loss(source, target)))

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
