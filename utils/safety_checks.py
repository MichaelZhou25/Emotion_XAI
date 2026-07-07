def check_no_overlap(train_subjects, val_subjects, test_subjects):
    train = set(map(int, train_subjects))
    val = set(map(int, val_subjects))
    test = set(map(int, test_subjects))
    assert train.isdisjoint(val), f'Train/val overlap: {train & val}'
    assert train.isdisjoint(test), f'Train/test overlap: {train & test}'
    assert val.isdisjoint(test), f'Val/test overlap: {val & test}'


def warn_legacy_protocol():
    msg = ('WARNING: legacy_loso uses the held-out target subject for checkpoint selection. '
           'Use only for paper-aligned comparison, not primary leakage-free generalization.')
    print('\n' + '=' * 90)
    print(msg)
    print('=' * 90 + '\n')
