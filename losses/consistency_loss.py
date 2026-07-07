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
