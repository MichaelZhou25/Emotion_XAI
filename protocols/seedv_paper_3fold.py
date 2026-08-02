from pathlib import Path

import numpy as np
import torch

from data.dataloader import build_index_fold_loaders
from graphs.affective_graph import get_affective_graph
from models import build_model
from trainer.engine import build_ema, build_lr_scheduler, build_optimizer
from trainer.evaluate import evaluate
from trainer.train_one_epoch import train_one_epoch
from utils.checkpoint import save_checkpoint
from utils.logger import append_csv, save_json
from utils.seed import set_seed


class SeedVPaperThreeFold:
    """Official-style subject-dependent three-fold evaluation for SEED-V.

    For every subject, each session's first, middle, and final five trials form
    the three folds. Corresponding groups are pooled across all three sessions.
    Two groups train the model and the remaining group is evaluated once after
    the fixed training schedule. This mirrors the split used by the original
    SEED-V papers while avoiding target-test model selection.
    """

    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.graph = get_affective_graph(
            cfg['dataset']['name'],
            cfg.get('graph', {}).get('name'),
        )
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available()
            and cfg['train'].get('device', 'auto') != 'cpu'
            else 'cpu'
        )
        self.save_dir = Path(cfg['logging']['save_dir'])

    def run(self):
        subjects = self.store.subjects
        target_subjects = (
            self.cfg.get('protocol', {}).get('target_subjects') or subjects
        )
        target_subjects = [int(subject) for subject in target_subjects]
        unknown = sorted(set(target_subjects) - set(subjects))
        if unknown:
            raise ValueError(f'Unknown SEED-V target subjects: {unknown}')

        fold_metrics = []
        subject_metrics = []
        for subject in target_subjects:
            per_subject = []
            for fold in range(3):
                metrics = self._run_fold(subject, fold)
                fold_metrics.append(metrics)
                per_subject.append(metrics)
            averaged = self._average_numeric(per_subject)
            averaged['subject_id'] = subject
            subject_metrics.append(averaged)
            save_json(
                self.save_dir / f'subject_{subject:02d}' / 'summary.json',
                averaged,
            )
        self._save_summary(fold_metrics, subject_metrics)

    def _run_fold(self, subject, fold):
        fold_seed = (
            int(self.cfg['train'].get('seed', 2026))
            + int(subject) * 10
            + int(fold)
        )
        set_seed(fold_seed)
        test_trials = list(range(fold * 5, fold * 5 + 5))
        train_trials = [
            trial for trial in range(15) if trial not in test_trials
        ]
        subject_mask = self.store.subject_id == int(subject)
        train_indices = np.where(
            subject_mask & np.isin(self.store.trial_id, train_trials)
        )[0]
        test_indices = np.where(
            subject_mask & np.isin(self.store.trial_id, test_trials)
        )[0]
        if len(np.intersect1d(train_indices, test_indices)):
            raise AssertionError('SEED-V paper folds overlap')
        sessions = sorted(
            (np.unique(self.store.session_id[subject_mask]) + 1)
            .astype(int)
            .tolist()
        )
        if sessions != [1, 2, 3]:
            raise ValueError(
                'SEED-V paper three-fold protocol requires sessions [1, 2, 3], '
                f'found {sessions} for subject {subject}'
            )

        fold_dir = (
            self.save_dir
            / f'subject_{subject:02d}'
            / f'fold_{fold + 1}'
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            fold_dir / 'split.json',
            {
                'protocol': 'seedv_paper_3fold',
                'subject_id': subject,
                'fold': fold + 1,
                'train_trial_positions_per_session': train_trials,
                'test_trial_positions_per_session': test_trials,
                'sessions': sessions,
                'train_samples': int(len(train_indices)),
                'test_samples': int(len(test_indices)),
                'model_selection': 'fixed_final_epoch',
                'test_used_for_selection': False,
                'fold_seed': fold_seed,
            },
        )
        train_loader, _, test_loader = build_index_fold_loaders(
            self.store,
            train_indices,
            test_indices,
            self.cfg,
        )
        model = build_model(self.cfg, self.graph).to(self.device)
        optimizer = build_optimizer(model, self.cfg)
        scheduler = build_lr_scheduler(optimizer, self.cfg)
        ema = build_ema(model, self.cfg)
        for epoch in range(1, self.cfg['train']['epochs'] + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                self.graph,
                self.cfg,
                self.device,
                epoch=epoch,
                ema=ema,
            )
            append_csv(
                fold_dir / 'train_log.csv',
                {'epoch': epoch, **train_metrics},
            )
            if scheduler is not None:
                scheduler.step()

        eval_model = ema.module if ema is not None else model
        test_metrics, outputs = evaluate(
            eval_model,
            test_loader,
            self.graph,
            self.cfg,
            self.device,
            return_outputs=True,
        )
        test_metrics.update({
            'subject_id': int(subject),
            'fold': int(fold + 1),
            'train_samples': int(len(train_indices)),
            'test_samples': int(len(test_indices)),
        })
        save_json(fold_dir / 'test_result.json', test_metrics)
        np.savez_compressed(
            fold_dir / 'predictions.npz',
            y_true=outputs['y_true'],
            y_pred=outputs['y_pred'],
            prob=outputs['prob'],
        )
        save_checkpoint(
            fold_dir / 'final_model.pth',
            eval_model,
            optimizer,
            self.cfg['train']['epochs'],
            test_metrics,
            extra={
                'model_source': 'ema' if ema is not None else 'online',
                'selection': 'fixed_final_epoch',
            },
        )
        print(
            f'[seedv-paper-3fold] subject={subject} fold={fold + 1} '
            f'test_acc={test_metrics["acc"]:.4f}'
        )
        return test_metrics

    @staticmethod
    def _average_numeric(metrics):
        keys = [
            key
            for key, value in metrics[0].items()
            if isinstance(value, (int, float))
            and key not in {'subject_id', 'fold'}
        ]
        return {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in keys
        }

    def _save_summary(self, fold_metrics, subject_metrics):
        metric_keys = [
            key
            for key, value in subject_metrics[0].items()
            if isinstance(value, (int, float))
            and key not in {'subject_id', 'fold'}
        ]
        summary = {
            'protocol': 'seedv_paper_3fold',
            'aggregation': 'mean within subject, then mean/std across subjects',
            'num_subjects': len(subject_metrics),
            'num_folds': len(fold_metrics),
            'classes': self.graph['classes'],
            'metrics': {
                key: {
                    'mean': float(
                        np.mean([metric[key] for metric in subject_metrics])
                    ),
                    'std': float(
                        np.std([metric[key] for metric in subject_metrics])
                    ),
                }
                for key in metric_keys
            },
            'confusion': np.sum(
                [
                    np.asarray(metric['confusion'], dtype=np.int64)
                    for metric in fold_metrics
                ],
                axis=0,
            ).tolist(),
        }
        save_json(self.save_dir / 'summary.json', summary)
        print(summary)
