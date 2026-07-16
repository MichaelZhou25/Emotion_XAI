# SEED-IV Session Results

These results use `HemiMVEAGLENet` without the Concept branch, cosine learning-rate decay, 5 warmup epochs, 100 epochs, and legacy test-selected LOSO. They are paper-aligned secondary results, not leakage-free model-selection results.

SEED-IV uses 10-second feature windows because some trials are too short for 30-second windows. Sessions 1, 2, and 3 contain 9,525, 9,240, and 9,090 windows, respectively. The combined dataset contains 27,855 windows. All runs have 15 subject-wise folds; in the combined run, every session from the target subject is held out together.

## Aggregate Results

| Data selection | Accuracy | Macro-F1 |
|---|---:|---:|
| Session 1 | 65.27 +/- 11.18 | 62.88 +/- 12.69 |
| Session 2 | 74.68 +/- 11.74 | 72.16 +/- 12.96 |
| Session 3 | 71.08 +/- 12.60 | 68.38 +/- 14.12 |
| Independent-session aggregate (45 folds) | 70.34 +/- 12.47 | 67.81 +/- 13.81 |
| Sessions 1+2+3 combined subject-wise LOSO | 53.48 +/- 11.10 | 51.54 +/- 11.98 |

## Per-Subject Accuracy

| Subject | Session 1 | Session 2 | Session 3 | Combined 123 |
|---:|---:|---:|---:|---:|
| 1 | 60.00 | 88.47 | 57.92 | 53.15 |
| 2 | 70.55 | 97.73 | 84.32 | 68.71 |
| 3 | 71.50 | 83.93 | 77.06 | 44.05 |
| 4 | 43.78 | 54.38 | 92.24 | 62.90 |
| 5 | 53.07 | 82.95 | 81.52 | 41.20 |
| 6 | 44.72 | 68.51 | 63.53 | 53.69 |
| 7 | 72.76 | 77.27 | 89.93 | 78.30 |
| 8 | 72.60 | 85.06 | 77.39 | 48.03 |
| 9 | 84.25 | 79.55 | 55.12 | 45.34 |
| 10 | 73.07 | 64.29 | 60.07 | 58.54 |
| 11 | 69.13 | 57.14 | 52.31 | 39.53 |
| 12 | 65.67 | 70.78 | 63.70 | 45.61 |
| 13 | 67.56 | 73.05 | 66.01 | 40.44 |
| 14 | 55.28 | 61.69 | 81.85 | 60.85 |
| 15 | 75.12 | 75.32 | 63.20 | 61.87 |

The earlier 67.28% session-1 result was a four-subject, 40-epoch probe and is not used in this table. The full session-1 result is 65.27% over all 15 subjects and 100 epochs.

The direct combined run is only MSHCL-like in using all three sessions; it is not an exact MSHCL reproduction. The large performance drop indicates that session-domain shift needs explicit modeling or session-balanced optimization rather than direct concatenation.
