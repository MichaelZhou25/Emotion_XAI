from pathlib import Path
import copy
import torch
from graphs.affective_graph import get_affective_graph
from data.splits import strict_loso_split
from data.dataloader import build_fold_loaders
from models import build_model
from trainer.engine import build_optimizer
from trainer.train_one_epoch import train_one_epoch
from trainer.evaluate import evaluate
from utils.logger import append_csv, save_json
from utils.checkpoint import save_checkpoint
from utils.safety_checks import check_no_overlap
from utils.seed import set_seed


class StrictDGLOSO:
    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.graph = get_affective_graph(cfg['dataset']['name'])
        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg['train'].get('device','auto') != 'cpu' else 'cpu')
        self.save_dir = Path(cfg['logging']['save_dir'])

    def run(self):
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
            train_sub, val_sub, test_sub = strict_loso_split(subjects, target, self.cfg['protocol'].get('val_ratio',0.2), self.cfg['train'].get('seed',2026))
            check_no_overlap(train_sub, val_sub, test_sub)
            save_json(fold_dir / 'split.json', {
                'protocol': 'strict_dg_loso', 'target_subject': int(target),
                'train_subjects': train_sub, 'val_subjects': val_sub, 'test_subjects': test_sub,
                'model_selection': 'source_val_acc', 'test_used_for_selection': False,
            })
            train_loader, val_loader, test_loader = build_fold_loaders(self.store, train_sub, val_sub, test_sub, self.cfg)
            model = build_model(self.cfg, self.graph).to(self.device)
            optimizer = build_optimizer(model, self.cfg)
            best_val = float('-inf')
            best_state = None
            wait = 0
            patience = self.cfg['train'].get('patience', 20)
            for epoch in range(1, self.cfg['train']['epochs'] + 1):
                train_m = train_one_epoch(model, train_loader, optimizer, self.graph, self.cfg, self.device, epoch=epoch)
                val_m, _ = evaluate(model, val_loader, self.graph, self.cfg, self.device)
                append_csv(fold_dir / 'train_log.csv', {'epoch': epoch, **train_m})
                append_csv(fold_dir / 'val_log.csv', {'epoch': epoch, **val_m})
                if val_m['acc'] > best_val:
                    best_val = val_m['acc']
                    best_state = copy.deepcopy(model.state_dict())
                    save_checkpoint(fold_dir / 'best_model.pth', model, optimizer, epoch, val_m)
                    wait = 0
                else:
                    wait += 1
                    if patience is not None and wait >= patience:
                        break
            model.load_state_dict(best_state)
            test_m, _ = evaluate(model, test_loader, self.graph, self.cfg, self.device)
            test_m['best_val_acc'] = float(best_val)
            save_json(fold_dir / 'test_result.json', test_m)
            all_metrics.append(test_m)
            print(f'[strict] target={target} test_acc={test_m["acc"]:.4f} best_val={best_val:.4f}')
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
