from pathlib import Path
import copy
import torch
from graphs.affective_graph import get_affective_graph
from data.splits import legacy_loso_split
from data.dataloader import build_fold_loaders
from models import build_model
from trainer.engine import build_ema, build_lr_scheduler, build_optimizer
from trainer.train_one_epoch import train_one_epoch
from trainer.evaluate import evaluate
from utils.logger import append_csv, save_json
from utils.checkpoint import save_checkpoint
from utils.safety_checks import warn_legacy_protocol
from utils.seed import set_seed


class LegacyLOSO:
    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.graph = get_affective_graph(
            cfg['dataset']['name'],
            cfg.get('graph', {}).get('name'),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg['train'].get('device','auto') != 'cpu' else 'cpu')
        self.save_dir = Path(cfg['logging']['save_dir'])

    def run(self):
        warn_legacy_protocol()
        all_metrics = []
        subjects = self.store.subjects
        target_subjects = self.cfg.get('protocol', {}).get('target_subjects') or subjects
        target_subjects = [int(s) for s in target_subjects]
        for fold_idx, target in enumerate(target_subjects):
            fold_seed = self._fold_seed(target, fold_idx)
            if fold_seed is not None:
                set_seed(fold_seed)
            fold_dir = self.save_dir / f'target_subject_{target:02d}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_sub, val_sub, test_sub = legacy_loso_split(subjects, target)
            save_json(fold_dir / 'split.json', {
                'protocol': 'legacy_loso', 'target_subject': int(target),
                'train_subjects': train_sub, 'val_subjects': [], 'test_subjects': test_sub,
                'model_selection': 'target_test_acc', 'test_used_for_selection': True,
                'fold_seed': fold_seed,
            })
            train_loader, _, test_loader = build_fold_loaders(self.store, train_sub, [], test_sub, self.cfg)
            model = build_model(self.cfg, self.graph).to(self.device)
            optimizer = build_optimizer(model, self.cfg)
            scheduler = build_lr_scheduler(optimizer, self.cfg)
            ema = build_ema(model, self.cfg)
            best_test = float('-inf')
            best_state = None
            for epoch in range(1, self.cfg['train']['epochs'] + 1):
                train_m = train_one_epoch(
                    model, train_loader, optimizer, self.graph, self.cfg, self.device,
                    epoch=epoch, ema=ema,
                )
                eval_model = ema.module if ema is not None else model
                test_m, _ = evaluate(eval_model, test_loader, self.graph, self.cfg, self.device)
                append_csv(fold_dir / 'train_log.csv', {'epoch': epoch, **train_m})
                append_csv(fold_dir / 'target_test_selection_log.csv', {'epoch': epoch, **test_m})
                if test_m['acc'] > best_test:
                    best_test = test_m['acc']
                    best_state = copy.deepcopy(eval_model.state_dict())
                    save_checkpoint(
                        fold_dir / 'best_model.pth', eval_model, optimizer, epoch, test_m,
                        extra={
                            'warning': 'selected_by_target_test_acc',
                            'model_source': 'ema' if ema is not None else 'online',
                        },
                    )
                if scheduler is not None:
                    scheduler.step()
            model.load_state_dict(best_state)
            final_m, _ = evaluate(model, test_loader, self.graph, self.cfg, self.device)
            final_m['best_target_test_acc_for_selection'] = float(best_test)
            save_json(fold_dir / 'test_result.json', final_m)
            all_metrics.append(final_m)
            print(f'[legacy] target={target} selected_test_acc={best_test:.4f}')
        self._save_summary(all_metrics)

    def _save_summary(self, metrics):
        import numpy as np
        keys = [k for k,v in metrics[0].items() if isinstance(v, (int,float))]
        summary = {k: {'mean': float(np.mean([m[k] for m in metrics])), 'std': float(np.std([m[k] for m in metrics]))} for k in keys}
        save_json(self.save_dir / 'summary.json', summary)
        print(summary)

    def _fold_seed(self, target, fold_idx):
        mode = self.cfg.get('protocol', {}).get('fold_seed_mode', 'global')
        if mode in (None, 'global'):
            return None
        base_seed = int(self.cfg['train'].get('seed', 2026))
        if mode == 'target':
            return base_seed + int(target)
        if mode == 'index':
            return base_seed + int(fold_idx)
        if mode == 'target_index':
            return base_seed + int(target) * 100 + int(fold_idx)
        raise ValueError(f'Unknown fold_seed_mode: {mode}')
