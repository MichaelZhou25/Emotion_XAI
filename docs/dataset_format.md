# Dataset Format

Recommended `.npz` cache:

- `x`: `[num_samples,T,62,5]`
- `y`: `[num_samples]`
- `subject_id`: `[num_samples]`, zero-based integer IDs
- `session_id`: `[num_samples]`
- `trial_id`: `[num_samples]`

SEED class order: `0=neutral, 1=positive, 2=negative`.

SEED-IV class order: `0=neutral, 1=happy, 2=sad, 3=fear`.
