from pathlib import Path
import pickle
import re

import numpy as np

from data.feature_window import build_temporal_windows, ensure_tcb
from data.session_selection import parse_sessions


# The official SEED-V feature archives use:
# 0=disgust, 1=fear, 2=sad, 3=neutral, 4=happy.
# EAGLE-Net keeps the SEED-IV-compatible class prefix and appends disgust:
# 0=neutral, 1=happy, 2=sad, 3=fear, 4=disgust.
SEEDV_OFFICIAL_TO_OURS = {3: 0, 4: 1, 2: 2, 1: 3, 0: 4}
SEEDV_NUM_TRIALS_PER_SESSION = 15
SEEDV_NUM_SESSIONS = 3


def subject_id_from_file(path):
    match = re.match(r'(\d+)_', Path(path).name)
    if match:
        return int(match.group(1)) - 1
    raise ValueError(f'Cannot determine SEED-V subject ID from {path}')


def _resolve_feature_dir(root_dir):
    root_dir = Path(root_dir)
    candidates = [
        root_dir,
        root_dir / 'EEG_DE_features',
        root_dir / 'SEED-V' / 'EEG_DE_features',
    ]
    for candidate in candidates:
        if any(candidate.glob('*_123.npz')):
            return candidate
    raise FileNotFoundError(
        f'No SEED-V *_123.npz feature files found under {root_dir}'
    )


def _unpickle_dict(value, field_name, source):
    value = np.asarray(value)
    payload = value.item() if value.shape == () else value.tobytes()
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError(f'{source}: {field_name} is not a pickled byte payload')
    decoded = pickle.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError(f'{source}: {field_name} did not decode to a dict')
    return decoded


def load_seedv_subject_file(path):
    """Load one official ``subject_123.npz`` file.

    The official files contain trusted pickled byte payloads rather than NumPy
    object arrays. Only use this loader with the official SEED-V release.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'data', 'label'}:
            raise ValueError(
                f'{path}: expected data/label fields, found {archive.files}'
            )
        data = _unpickle_dict(archive['data'], 'data', path)
        labels = _unpickle_dict(archive['label'], 'label', path)
    expected_keys = set(range(SEEDV_NUM_SESSIONS * SEEDV_NUM_TRIALS_PER_SESSION))
    if set(data) != expected_keys or set(labels) != expected_keys:
        raise ValueError(f'{path}: expected trial keys 0..44')
    return data, labels


def prepare_seedv_features(
    root_dir,
    out_path,
    time_steps=10,
    stride=1,
    session=1,
    input_type='de',
    num_channels=62,
    num_bands=5,
    sessions=None,
):
    if input_type.lower() not in {'de', 'official_de'}:
        raise ValueError(
            'Official SEED-V *.npz files contain DE features only; '
            f'unsupported input_type={input_type!r}'
        )
    selected_sessions = parse_sessions(sessions if sessions is not None else session)
    feature_dir = _resolve_feature_dir(root_dir)
    feature_files = sorted(
        feature_dir.glob('*_123.npz'),
        key=subject_id_from_file,
    )
    if len(feature_files) != 16:
        raise FileNotFoundError(
            f'Expected 16 SEED-V subject feature files in {feature_dir}, '
            f'found {len(feature_files)}'
        )

    all_x, all_y, all_sub, all_sess, all_trial = [], [], [], [], []
    for feature_file in feature_files:
        subject_id = subject_id_from_file(feature_file)
        data, labels = load_seedv_subject_file(feature_file)
        for selected_session in selected_sessions:
            session_offset = (
                selected_session - 1
            ) * SEEDV_NUM_TRIALS_PER_SESSION
            for trial_id in range(SEEDV_NUM_TRIALS_PER_SESSION):
                official_trial_id = session_offset + trial_id
                trial_feature = ensure_tcb(
                    data[official_trial_id],
                    num_channels,
                    num_bands,
                )
                trial_labels = np.asarray(labels[official_trial_id]).reshape(-1)
                if trial_labels.size != trial_feature.shape[0]:
                    raise ValueError(
                        f'{feature_file}: trial {official_trial_id} has '
                        f'{trial_feature.shape[0]} samples but '
                        f'{trial_labels.size} labels'
                    )
                unique_labels = np.unique(trial_labels.astype(np.int64))
                if len(unique_labels) != 1:
                    raise ValueError(
                        f'{feature_file}: trial {official_trial_id} has '
                        f'non-constant labels {unique_labels.tolist()}'
                    )
                official_label = int(unique_labels[0])
                if official_label not in SEEDV_OFFICIAL_TO_OURS:
                    raise ValueError(
                        f'{feature_file}: unsupported label {official_label}'
                    )
                label = SEEDV_OFFICIAL_TO_OURS[official_label]
                x, y = build_temporal_windows(
                    trial_feature,
                    label,
                    time_steps,
                    stride,
                )
                if len(y) == 0:
                    continue
                all_x.append(x)
                all_y.append(y)
                all_sub.append(np.full(len(y), subject_id, dtype=np.int64))
                all_sess.append(
                    np.full(len(y), selected_session - 1, dtype=np.int64)
                )
                all_trial.append(
                    np.full(len(y), trial_id, dtype=np.int64)
                )

    if not all_x:
        raise ValueError(
            f'No SEED-V windows were produced with time_steps={time_steps}'
        )
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
