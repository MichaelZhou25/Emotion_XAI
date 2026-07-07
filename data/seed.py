from pathlib import Path
import re
import numpy as np
import scipy.io as sio
from data.feature_window import ensure_tcb, build_temporal_windows, trial_key_sort_key

# Official SEED coarse labels are often represented as positive=1, neutral=0, negative=-1.
# We use class order: neutral=0, positive=1, negative=2.
SEED_TRIAL_LABELS_ORIGINAL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
SEED_LABEL_MAP = {0: 0, 1: 1, -1: 2}
SEED_TRIAL_LABELS = [SEED_LABEL_MAP[x] for x in SEED_TRIAL_LABELS_ORIGINAL]


def subject_id_from_file(path):
    m = re.match(r'(\d+)_', Path(path).name)
    if m:
        return int(m.group(1)) - 1
    nums = re.findall(r'\d+', Path(path).stem)
    return int(nums[0]) - 1 if nums else 0


def prepare_seed_features(root_dir, out_path, time_steps=10, stride=1, session=1, input_type='lds_de', num_channels=62, num_bands=5):
    root_dir = Path(root_dir)
    session_dir = root_dir / str(session)
    search_dir = session_dir if session_dir.exists() else root_dir
    mat_files = sorted(search_dir.glob('*.mat'), key=lambda p: subject_id_from_file(p))
    if not mat_files:
        raise FileNotFoundError(f'No .mat feature files found in {search_dir}')
    key_prefix = 'de_LDS' if input_type.lower() in ['lds_de', 'de_lds'] else 'de'
    all_x, all_y, all_sub, all_sess, all_trial = [], [], [], [], []
    for f in mat_files:
        mat = sio.loadmat(f, verify_compressed_data_integrity=False)
        keys = [k for k in mat.keys() if k.startswith(key_prefix)]
        keys = sorted(keys, key=trial_key_sort_key)
        sid = subject_id_from_file(f)
        for trial_idx, k in enumerate(keys[:len(SEED_TRIAL_LABELS)]):
            tcb = ensure_tcb(mat[k], num_channels, num_bands)
            x, y = build_temporal_windows(tcb, SEED_TRIAL_LABELS[trial_idx], time_steps, stride)
            all_x.append(x); all_y.append(y)
            all_sub.append(np.full(len(y), sid, dtype=np.int64))
            all_sess.append(np.full(len(y), int(session)-1, dtype=np.int64))
            all_trial.append(np.full(len(y), trial_idx, dtype=np.int64))
    arrays = {
        'x': np.concatenate(all_x).astype(np.float32),
        'y': np.concatenate(all_y).astype(np.int64),
        'subject_id': np.concatenate(all_sub).astype(np.int64),
        'session_id': np.concatenate(all_sess).astype(np.int64),
        'trial_id': np.concatenate(all_trial).astype(np.int64),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return arrays
