class BestMetricTracker:
    def __init__(self, mode='max'):
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = -1

    def update(self, value, epoch):
        improved = value > self.best if self.mode == 'max' else value < self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
        return improved
