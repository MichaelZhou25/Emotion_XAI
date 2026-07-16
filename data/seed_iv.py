from pathlib import Path
import re
import numpy as np
import scipy.io as sio
from data.feature_window import ensure_tcb, build_temporal_windows, trial_key_sort_key
from data.session_selection import parse_sessions

# SEED-IV public feature labels commonly use 0=neutral, 1=sad, 2=fear, 3=happy.
# We use class order: 0=neutral, 1=happy, 2=sad, 3=fear.
SEEDIV_OFFICIAL_TO_OURS = {0: 0, 3: 1, 1: 2, 2: 3}
SEEDIV_LABELS_BY_SESSION_OFFICIAL = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
]
SEEDIV_LABELS_BY_SESSION = [[SEEDIV_OFFICIAL_TO_OURS[x] for x in s] for s in SEEDIV_LABELS_BY_SESSION_OFFICIAL]


def subject_id_from_file(path):
    m = re.match(r'(\d+)_', Path(path).name)
    if m:
        return int(m.group(1)) - 1
    nums = re.findall(r'\d+', Path(path).stem)
    return int(nums[0]) - 1 if nums else 0


def prepare_seediv_features(root_dir, out_path, time_steps=10, stride=1, session=1, input_type='lds_de', num_channels=62, num_bands=5, sessions=None):
    root_dir = Path(root_dir)
    selected_sessions = parse_sessions(sessions if sessions is not None else session)
    key_prefix = 'de_LDS' if input_type.lower() in ['lds_de', 'de_lds'] else 'de'
    all_x, all_y, all_sub, all_sess, all_trial = [], [], [], [], []
    for selected_session in selected_sessions:
        session_dir = root_dir / str(selected_session)
        if session_dir.exists():
            search_dir = session_dir
        elif len(selected_sessions) == 1:
            search_dir = root_dir
        else:
            raise FileNotFoundError(
                f'Multi-session loading requires session subdirectories; missing {session_dir}'
            )
        mat_files = sorted(search_dir.glob('*.mat'), key=lambda p: subject_id_from_file(p))
        if not mat_files:
            raise FileNotFoundError(f'No .mat feature files found in {search_dir}')
        labels = SEEDIV_LABELS_BY_SESSION[selected_session - 1]
        for f in mat_files:
            mat = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = [k for k in mat.keys() if k.startswith(key_prefix)]
            keys = sorted(keys, key=trial_key_sort_key)
            sid = subject_id_from_file(f)
            for trial_idx, k in enumerate(keys[:len(labels)]):
                tcb = ensure_tcb(mat[k], num_channels, num_bands)
                x, y = build_temporal_windows(tcb, labels[trial_idx], time_steps, stride)
                all_x.append(x); all_y.append(y)
                all_sub.append(np.full(len(y), sid, dtype=np.int64))
                all_sess.append(np.full(len(y), selected_session - 1, dtype=np.int64))
                all_trial.append(np.full(len(y), trial_idx, dtype=np.int64))
    arrays = {
        'x': np.concatenate(all_x).astype(np.float32, copy=False),
        'y': np.concatenate(all_y).astype(np.int64, copy=False),
        'subject_id': np.concatenate(all_sub).astype(np.int64, copy=False),
        'session_id': np.concatenate(all_sess).astype(np.int64, copy=False),
        'trial_id': np.concatenate(all_trial).astype(np.int64, copy=False),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return arrays
