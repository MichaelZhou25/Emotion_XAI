# Reproducibility Notes

Every fold saves:

- `split.json`
- `train_log.csv`
- `val_log.csv` for strict protocol
- `target_test_selection_log.csv` for legacy protocol
- `best_model.pth`
- `test_result.json`

Always report whether `test_used_for_selection` is true or false.
