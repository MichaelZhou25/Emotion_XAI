import math

import torch
import torch.nn.functional as F


def branch_consistency_loss(outputs, enabled_keys=None):
    if enabled_keys is None:
        enabled_keys = ['logits_direct', 'logits_proto', 'logits_edge', 'logits_concept']
    p_teacher = F.softmax(outputs['logits_final'].detach(), dim=-1)
    loss = 0.0
    count = 0
    for k in enabled_keys:
        if k not in outputs:
            continue
        logp = F.log_softmax(outputs[k], dim=-1)
        loss = loss + F.kl_div(logp, p_teacher, reduction='batchmean')
        count += 1
    return loss / max(count, 1)


def path_direct_js_loss(outputs, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    path_log_prob = F.log_softmax(outputs['logits_edge'] / temperature, dim=-1)
    direct_log_prob = F.log_softmax(outputs['logits_direct'] / temperature, dim=-1)
    mixture_log_prob = torch.logsumexp(
        torch.stack([path_log_prob, direct_log_prob], dim=0),
        dim=0,
    ) - math.log(2.0)
    path_prob = path_log_prob.exp()
    direct_prob = direct_log_prob.exp()
    js = 0.5 * (
        (path_prob * (path_log_prob - mixture_log_prob)).sum(dim=-1)
        + (direct_prob * (direct_log_prob - mixture_log_prob)).sum(dim=-1)
    )
    return js.mean() * temperature ** 2
