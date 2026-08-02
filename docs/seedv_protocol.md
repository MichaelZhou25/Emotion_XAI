# SEED-V evaluation protocol

## Active project default

The active SEED-V protocol is the original paper-aligned, subject-dependent
three-fold evaluation. Run it with:

```powershell
python train.py --config configs/seedv.yaml
```

Its current headline result is **54.09+/-9.01% accuracy**. The session-wise
strict cross-subject LOSO result is retained below only as a harder diagnostic
and is not used as the project's main SEED-V score.

## Data

EAGLE-Net uses the official `EEG_DE_features/*_123.npz` files: 16 subjects,
three sessions, 15 trials per session, 62 EEG channels, and five DE frequency
bands. Official labels are remapped to the project order:

`neutral=0, happy=1, sad=2, fear=3, disgust=4`.

Temporal windows are constructed inside individual trials, so no window can
cross a trial or emotion boundary.

## Paper-aligned subject-dependent evaluation

The original SEED-V literature uses three-fold cross-validation for every
subject. Each session is divided into the first, middle, and final five trials;
the corresponding groups from all three sessions are pooled. Two groups train
the model and the held-out group is tested.

Use:

```powershell
python train.py --config configs/seedv_paper_3fold_v8_full100.yaml
```

The implementation uses a fixed final epoch and evaluates the held-out fold
once. This avoids the target-test epoch selection used by some legacy deep
learning evaluations.

The original 2019 work reports EEG-only accuracy around 70.8% with a
subject-dependent classifier. Later DE-only work reports approximately
69.50+/-10.28% under a related three-fold setup. These values are context, not
direct architecture-matched baselines: the current model uses 10-second
within-trial DE sequences, whereas classical baselines classify individual DE
samples.

Primary references:

- Li et al., "Classification of Five Emotions from EEG and Eye Movement
  Signals: Discrimination Ability and Stability over Time," NER 2019:
  https://doi.org/10.1109/NER.2019.8716943
- Wu et al., "Investigating EEG-Based Functional Connectivity Patterns for
  Multimodal Emotion Recognition":
  https://arxiv.org/abs/2004.01973

## Strict subject-independent evaluation

For deployment-style generalization to unseen people, use leakage-free LOSO:

```powershell
python train.py --config configs/seedv_strict_dg_loso_v8_full100.yaml
```

In every fold one subject is held out for testing and source subjects are split
again for validation. The target subject is never used for model selection.
These results must not be compared directly with the subject-dependent
three-fold numbers above.

## EAGLE-Net V8 result

Run date: 2026-07-30. Configuration:
`configs/seedv_paper_3fold_v8_full100.yaml`.

The experiment completed all 16 subjects, 3 folds per subject, and 100 fixed
epochs per fold. Metrics are first averaged over the three folds of each
subject and then summarized across 16 subjects:

| Metric | Mean | Subject SD |
| --- | ---: | ---: |
| Accuracy | 54.09% | 9.01% |
| Balanced accuracy | 55.95% | 8.83% |
| Macro F1 | 51.92% | 9.24% |
| Cohen's kappa | 0.428 | 0.103 |

The pooled-sample accuracy is 54.75%. Pooled per-class recall is:

| Emotion | Recall |
| --- | ---: |
| Neutral | 61.45% |
| Happy | 64.01% |
| Sad | 44.84% |
| Fear | 66.40% |
| Disgust | 39.47% |

The strongest subject-average accuracy is 66.56% and the weakest is 36.60%.
Training accuracy frequently reaches 100%, including a severe failure fold
with 3.27% test accuracy. This indicates overfitting and stimulus-group shift,
not a label conversion error: every fold contains all five labels and the
conversion was independently validated.

The current result is 16.71 percentage points below the original paper's
70.8% EEG-only result. It is a valid first application of the current V8
architecture, but not yet a tuned SEED-V result. The most important next
experiment is source-only validation or nested validation for regularization
and epoch selection; selecting epochs on the held-out test fold must remain
disallowed.

Complete results, predictions, and checkpoints are under:
`results/seed_v/paper_3fold/v8_full100/`.

## Session-wise strict cross-subject result

Run date: 2026-07-30. Configuration:
`configs/seedv_strict_dg_loso_v8_full100.yaml`.

To match the project's SEED and SEED-IV comparison convention, each session is
evaluated independently with 16-fold strict LOSO. Source subjects are split
again for model-selection validation; the target subject is evaluated only
after selection.

| Session | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| 1 | 31.62+/-10.15% | 30.64+/-9.84% | 20.60+/-10.32% |
| 2 | 26.80+/-9.09% | 25.95+/-10.13% | 14.53+/-10.79% |
| 3 | 28.65+/-9.49% | 26.41+/-11.35% | 17.07+/-11.66% |
| Three-session mean | 29.02% | 27.66% | 17.40% |

Across all 48 folds, accuracy is 29.02+/-9.79%. The pooled-sample accuracy is
29.26%. Per-class recall is 28.39% neutral, 24.94% happy, 43.96% sad, 15.24%
fear, and 27.00% disgust. The model overpredicts sad, the largest class in the
windowed data, and poorly transfers fear-related edge decisions to unseen
subjects.

This result is above the 20% five-class chance level but is not competitive
with dedicated cross-subject SEED-V methods. It shows that the current
edge-only V8 architecture needs class balancing and stronger cross-subject
alignment before it can be used as the project's SEED-V result.
