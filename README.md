# EAGLE-Net

**EAGLE-Net: Explainable Affective Geometry and Label-conditioned Evidence Network**

This project implements a domain-generalized and explainable EEG emotion recognition framework for SEED and SEED-IV using extracted DE/LDS-DE features.

## Key Design

- **Input:** extracted DE/LDS-DE features, reshaped as `[N, T, C, B]`.
- **Datasets:** SEED and SEED-IV.
- **Primary protocol:** `strict_dg_loso`, leakage-free source-only domain generalization.
- **Secondary protocol:** `legacy_loso`, paper-aligned test-selected LOSO for comparison with prior open-source implementations.
- **Model:** shared EEG encoder + direct auxiliary head + hyperbolic prototype branch + edge-conditioned evidence branch + concept branch.
- **XAI:** edge attention maps, prototype distances, concept scores, hemispheric evidence, deletion/insertion, stability, sanity-check utilities.

## Recommended Usage

```bash
pip install -r requirements.txt
python train.py --config configs/seed_strict.yaml
python train.py --config configs/seed_legacy.yaml
python train.py --config configs/seediv_strict.yaml
python train.py --config configs/seediv_legacy.yaml
```

## Data Format

The preferred preprocessed cache is `.npz`:

```python
x:          [num_samples, T, 62, 5]
y:          [num_samples]
subject_id: [num_samples]
session_id: [num_samples]
trial_id:   [num_samples]
```

If no processed cache is available, the provided preprocessing scripts can read SEED/SEED-IV extracted `.mat` feature files containing `de_LDS*` or `de*` keys.

## Session Selection

SEED and SEED-IV configs may select one session or combine all three sessions:

```yaml
dataset:
  sessions: [1]       # change to [2], [3], or [1, 2, 3]
```

The same selection can be overridden without editing a config. The CLI override uses a matching automatic cache name and a separate result subdirectory:

```bash
python train.py --config configs/optimization/seed_no_concept_cosine_full100.yaml --sessions 1
python train.py --config configs/optimization/seed_no_concept_cosine_full100.yaml --sessions 2
python train.py --config configs/optimization/seed_no_concept_cosine_full100.yaml --sessions 3
python train.py --config configs/optimization/seed_no_concept_cosine_full100.yaml --sessions 123
```

In `123` mode, LOSO remains subject-wise: all three sessions of the held-out subject are assigned to the test set. See the [SEED report](docs/seed_session_results.md) and [SEED-IV report](docs/seediv_session_results.md) for full100 results and protocol caveats.

## Protocols

### Strict DG-LOSO

Target subject is never used for training, validation, checkpoint selection, early stopping, or hyperparameter selection. The target subject is evaluated once after loading the best source-validation checkpoint.

### Legacy LOSO

Target subject is evaluated each epoch and used for checkpoint selection, following several historical open-source EEG emotion-recognition implementations. This is provided only for paper-aligned comparison and is not the primary leakage-free generalization result.

## Smoke Test

Set `dataset.synthetic.enabled: true` in a config to run without real SEED files.
