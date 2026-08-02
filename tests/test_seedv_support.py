import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.seed_v import prepare_seedv_features
from graphs.affective_graph import get_affective_graph
from losses.eagle_loss import compute_eagle_loss
from models import build_model
from utils.config import load_config


class SeedVSupportTests(unittest.TestCase):
    def test_official_feature_conversion_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'EEG_DE_features'
            root.mkdir()
            for subject in range(1, 17):
                data = {
                    trial: np.full(
                        (1, 310),
                        subject + trial / 100.0,
                        dtype=np.float64,
                    )
                    for trial in range(45)
                }
                labels = {
                    trial: np.asarray([trial % 5], dtype=np.float64)
                    for trial in range(45)
                }
                np.savez(
                    root / f'{subject}_123.npz',
                    data=pickle.dumps(data),
                    label=pickle.dumps(labels),
                )
            out = Path(tmp) / 'cache.npz'
            arrays = prepare_seedv_features(
                root,
                out,
                time_steps=1,
                sessions=[1, 2, 3],
            )

            self.assertEqual(arrays['x'].shape, (16 * 45, 1, 62, 5))
            self.assertEqual(sorted(np.unique(arrays['y']).tolist()), [0, 1, 2, 3, 4])
            self.assertEqual(sorted(np.unique(arrays['subject_id']).tolist()), list(range(16)))
            self.assertEqual(sorted(np.unique(arrays['session_id']).tolist()), [0, 1, 2])
            self.assertEqual(sorted(np.unique(arrays['trial_id']).tolist()), list(range(15)))
            self.assertTrue(out.exists())

    def test_seedv_graph_and_current_model_loss(self):
        cfg = load_config('configs/seedv_paper_3fold_v8_probe.yaml')
        cfg['dataset']['time_steps'] = 2
        cfg['model']['window_size'] = 2
        cfg['model']['d_model'] = 16
        cfg['model']['d_proto'] = 8
        graph = get_affective_graph(
            'SEED-V',
            'seedv_neutral_centered_edge_graph',
        )
        model = build_model(cfg, graph)
        labels = torch.arange(5)
        outputs = model(torch.randn(5, 2, 62, 5))
        loss, _ = compute_eagle_loss(
            outputs,
            labels,
            graph,
            cfg,
            subject_ids=torch.zeros(5, dtype=torch.long),
            epoch=1,
        )

        self.assertEqual(graph['classes'], ['neutral', 'happy', 'sad', 'fear', 'disgust'])
        self.assertEqual(outputs['logits_final'].shape, (5, 5))
        self.assertEqual(outputs['edge_weights'].shape, (5, 6))
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()
