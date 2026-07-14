def scheduled_value(base, epoch=None, schedule='constant', warmup_epochs=0, ramp_epochs=0):
    base = float(base)
    if epoch is None or schedule in (None, 'constant'):
        return base
    if schedule != 'linear':
        raise ValueError(f'Unknown schedule: {schedule}')

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base
    scale = min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)))
    return base * scale
