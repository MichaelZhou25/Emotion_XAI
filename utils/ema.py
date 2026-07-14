import copy
import math

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, warmup_updates=100):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay
        if self.warmup_updates > 0:
            decay *= 1.0 - math.exp(-self.updates / float(self.warmup_updates))

        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)
