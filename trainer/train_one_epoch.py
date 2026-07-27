import torch
import torch.nn.functional as F
from tqdm import tqdm
from losses.eagle_loss import compute_eagle_loss
from losses.domain_alignment_loss import linear_mmd_loss
from utils.schedules import scheduled_value


def _set_grl_progress(model, cfg, epoch):
    model_cfg = cfg.get('model', {})
    value = scheduled_value(
        model_cfg.get('grl_lambda', 0.2),
        epoch,
        model_cfg.get('grl_schedule', 'constant'),
        model_cfg.get('grl_warmup_epochs', 0),
        model_cfg.get('grl_ramp_epochs', 0),
    )
    target = model.module if hasattr(model, 'module') else model
    if hasattr(target, 'set_grl_lambda'):
        target.set_grl_lambda(value)
    return value


def _merge_target_domain_loss(
    total_loss,
    loss_dict,
    target_domain_loss,
    cfg,
    epoch=None,
    target_feature_alignment_loss=None,
):
    adaptation_cfg = cfg.get('train', {}).get('target_adaptation', {})
    target_weight = scheduled_value(
        adaptation_cfg.get('target_domain_weight', 1.0),
        epoch,
        adaptation_cfg.get('schedule', 'constant'),
        adaptation_cfg.get('warmup_epochs', 0),
        adaptation_cfg.get('ramp_epochs', 0),
    )
    source_domain_loss = loss_dict['domain']
    joint_domain_loss = (
        source_domain_loss + target_weight * target_domain_loss
    ) / (1.0 + target_weight)
    if target_feature_alignment_loss is not None:
        feature_weight = float(
            adaptation_cfg.get('feature_alignment_weight', 0.0)
        )
        joint_domain_loss = (
            joint_domain_loss
            + target_weight
            * feature_weight
            * target_feature_alignment_loss
        )
    lambda_domain = float(cfg.get('loss', {}).get('lambda_domain', 0.0))
    total_loss = (
        total_loss
        + lambda_domain * (joint_domain_loss - source_domain_loss)
    )
    loss_dict = dict(loss_dict)
    loss_dict['domain'] = joint_domain_loss
    if target_feature_alignment_loss is not None:
        loss_dict['target_feature_alignment'] = (
            target_feature_alignment_loss
        )
    loss_dict['total'] = total_loss
    return total_loss, loss_dict


def _target_domain_weight(cfg, epoch=None):
    adaptation_cfg = cfg.get('train', {}).get('target_adaptation', {})
    return scheduled_value(
        adaptation_cfg.get('target_domain_weight', 1.0),
        epoch,
        adaptation_cfg.get('schedule', 'constant'),
        adaptation_cfg.get('warmup_epochs', 0),
        adaptation_cfg.get('ramp_epochs', 0),
    )


def _target_feature_alignment_loss(source_outputs, target_outputs, cfg):
    adaptation_cfg = cfg.get('train', {}).get('target_adaptation', {})
    name = adaptation_cfg.get('feature_alignment', 'none')
    if name in (None, 'none'):
        return None
    if name == 'linear_mmd':
        return linear_mmd_loss(
            source_outputs['z_fused'],
            target_outputs['z_fused'],
        )
    raise ValueError(f'Unknown target feature alignment: {name}')


def train_one_epoch(
    model,
    loader,
    optimizer,
    graph,
    cfg,
    device,
    epoch=None,
    ema=None,
    target_loader=None,
):
    model.train()
    grl_lambda = _set_grl_progress(model, cfg, epoch)
    adaptation_enabled = bool(
        cfg.get('train', {})
        .get('target_adaptation', {})
        .get('enabled', False)
    )
    target_domain_weight = (
        _target_domain_weight(cfg, epoch)
        if adaptation_enabled
        else 0.0
    )
    adaptation_active = adaptation_enabled and target_domain_weight > 0
    if adaptation_enabled and target_loader is None:
        raise ValueError(
            'target_adaptation requires an unlabeled target loader'
        )
    target_iterator = iter(target_loader) if adaptation_active else None
    total_loss = 0.0
    total = 0
    correct = 0
    domain_correct = 0
    domain_total = 0
    loss_sums = {}
    for batch in tqdm(loader, desc='train', leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        subject_id = batch.get('subject_id')
        subject_id = subject_id.to(device) if subject_id is not None else None
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss, loss_dict = compute_eagle_loss(outputs, y, graph, cfg, subject_ids=subject_id, epoch=epoch)
        target_domain_correct = 0
        target_domain_total = 0
        if adaptation_active:
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch = next(target_iterator)
            target_x = target_batch['x'].to(device)
            target_subject_id = target_batch['subject_id'].to(device)
            target_outputs = model(target_x)
            target_domain_loss = F.cross_entropy(
                target_outputs['domain_logits'],
                target_subject_id,
            )
            target_feature_alignment_loss = (
                _target_feature_alignment_loss(
                    outputs,
                    target_outputs,
                    cfg,
                )
            )
            loss, loss_dict = _merge_target_domain_loss(
                loss,
                loss_dict,
                target_domain_loss,
                cfg,
                epoch=epoch,
                target_feature_alignment_loss=(
                    target_feature_alignment_loss
                ),
            )
            target_domain_prediction = target_outputs[
                'domain_logits'
            ].argmax(dim=-1)
            target_domain_correct = int(
                (target_domain_prediction == target_subject_id).sum()
            )
            target_domain_total = target_subject_id.shape[0]
        loss.backward()
        if cfg['train'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
        optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        total += bs
        total_loss += float(loss.detach()) * bs
        pred = outputs['logits_final'].argmax(dim=-1)
        correct += int((pred == y).sum())
        if subject_id is not None and cfg.get('model', {}).get('use_domain_adversarial', False) and 'domain_logits' in outputs:
            domain_pred = outputs['domain_logits'].argmax(dim=-1)
            domain_correct += (
                int((domain_pred == subject_id).sum())
                + target_domain_correct
            )
            domain_total += bs + target_domain_total
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach()) * bs
    metrics = {
        'loss': total_loss / max(total,1),
        'acc': correct / max(total,1),
        'lr': optimizer.param_groups[0]['lr'],
        'grl_lambda': grl_lambda,
        'target_domain_weight': target_domain_weight,
    }
    if domain_total:
        metrics['domain_acc'] = domain_correct / max(domain_total, 1)
    for k, v in loss_sums.items():
        metrics[f'loss_{k}'] = v / max(total,1)
    return metrics
