# SEED Session Results

These results use `HemiMVEAGLENet` without the Concept branch, cosine learning-rate decay, 5 warmup epochs, 100 epochs, and legacy test-selected LOSO. They are paper-aligned secondary results, not leakage-free model-selection results.

Each individual session contains 44,385 windows. The combined dataset contains 133,155 windows. All runs have 15 subject-wise folds; the combined run holds out all 8,877 windows from the target subject across sessions 1, 2, and 3.

## Aggregate Results

| Data selection | Accuracy | Macro-F1 |
|---|---:|---:|
| Session 1 | 82.71 +/- 5.57 | 82.30 +/- 5.86 |
| Session 2 | 77.59 +/- 8.59 | 75.67 +/- 10.16 |
| Session 3 | 78.27 +/- 10.40 | 77.60 +/- 10.67 |
| Independent-session aggregate (45 folds) | 79.52 +/- 8.73 | 78.52 +/- 9.57 |
| Sessions 1+2+3 combined subject-wise LOSO | 69.84 +/- 6.59 | 68.67 +/- 7.00 |

## Per-Subject Accuracy

| Subject | Session 1 | Session 2 | Session 3 | Combined 123 |
|---:|---:|---:|---:|---:|
| 1 | 77.53 | 83.68 | 80.03 | 61.48 |
| 2 | 85.30 | 92.06 | 75.90 | 71.60 |
| 3 | 83.24 | 79.86 | 79.62 | 63.14 |
| 4 | 94.53 | 77.09 | 53.46 | 70.77 |
| 5 | 84.96 | 75.40 | 78.64 | 66.55 |
| 6 | 71.51 | 72.42 | 84.79 | 69.37 |
| 7 | 88.10 | 65.83 | 83.54 | 76.47 |
| 8 | 82.22 | 83.68 | 91.04 | 81.36 |
| 9 | 80.57 | 84.62 | 78.17 | 71.40 |
| 10 | 82.05 | 74.05 | 68.94 | 65.63 |
| 11 | 91.72 | 82.05 | 85.67 | 82.62 |
| 12 | 79.38 | 65.43 | 68.27 | 60.50 |
| 13 | 79.62 | 69.92 | 72.22 | 62.50 |
| 14 | 77.70 | 65.50 | 73.84 | 69.76 |
| 15 | 82.26 | 92.23 | 99.86 | 74.48 |

The combined protocol is closer to studies that use all three sessions, but it is not automatically an exact MSHCL reproduction. Matching another paper also requires the same windowing, normalization, checkpoint selection, and session treatment. The large drop in the direct combined run indicates that session shift needs explicit handling; simply concatenating sessions is not a performance improvement.
