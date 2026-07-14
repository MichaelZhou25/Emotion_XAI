from pathlib import Path
import copy
import yaml


def deep_update(base, override):
    base = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    extends = cfg.pop('extends', None)
    if extends is None:
        return cfg
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_update(load_config(base_path), cfg)


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
