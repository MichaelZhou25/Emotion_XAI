from losses.classification_loss import ce_loss
from losses.edge_loss import edge_bce_loss
from losses.concept_loss import concept_mse_loss
from losses.consistency_loss import branch_consistency_loss
from losses.hierarchical_path_loss import (
    hierarchical_child_loss,
    hierarchical_root_loss,
)
from losses.hyperbolic_contrastive_loss import (
    hyperbolic_class_margin_loss,
    hyperbolic_hierarchy_margin_loss,
    hyperbolic_hierarchy_prototype_nce_loss,
    hyperbolic_prototype_nce_loss,
    hyperbolic_subject_centroid_loss,
    hyperbolic_supcon_loss,
)
from losses.radius_loss import radius_hierarchy_loss
from utils.schedules import scheduled_value


def _scheduled_weight(w, prefix, epoch):
    base = float(w.get(f'lambda_{prefix}', 0.0))
    is_hp = prefix.startswith('hp_')
    schedule = w.get(f'{prefix}_schedule', w.get('hp_schedule', 'constant') if is_hp else 'constant')
    warmup = int(w.get(f'{prefix}_warmup_epochs', w.get('hp_warmup_epochs', 0) if is_hp else 0))
    ramp = int(w.get(f'{prefix}_ramp_epochs', w.get('hp_ramp_epochs', 0) if is_hp else 0))
    return scheduled_value(base, epoch, schedule, warmup, ramp)


def compute_eagle_loss(outputs, labels, graph, cfg, subject_ids=None, epoch=None):
    w = cfg.get('loss', {})
    model_cfg = cfg.get('model', {})
    loss_dict = {}
    hp_proto_nce_weight = _scheduled_weight(w, 'hp_proto_nce', epoch)
    hp_supcon_weight = _scheduled_weight(w, 'hp_supcon', epoch)
    hp_margin_weight = _scheduled_weight(w, 'hp_margin', epoch)
    hp_subject_centroid_weight = _scheduled_weight(w, 'hp_subject_centroid', epoch)
    domain_weight = _scheduled_weight(w, 'domain', epoch)
    label_smoothing = cfg.get('train', {}).get('label_smoothing', 0.0)

    loss_dict['ce_final'] = ce_loss(outputs['logits_final'], labels, label_smoothing)
    loss_dict['ce_direct'] = ce_loss(outputs['logits_direct'], labels, label_smoothing) if model_cfg.get('use_direct_head', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['ce_proto'] = ce_loss(outputs['logits_proto'], labels, label_smoothing) if model_cfg.get('use_proto', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['ce_path'] = ce_loss(outputs['logits_edge'], labels, label_smoothing) if w.get('lambda_path', 0.0) > 0 else outputs['logits_final'].new_tensor(0.0)
    loss_dict['hier_root'] = hierarchical_root_loss(
        outputs, labels, graph, label_smoothing,
    ) if w.get('lambda_hier_root', 0.0) > 0 else outputs['logits_final'].new_tensor(0.0)
    loss_dict['hier_child'] = hierarchical_child_loss(
        outputs,
        labels,
        graph,
        label_smoothing=label_smoothing,
        balance=w.get('hier_child_balance', True),
        max_weight=w.get('hier_child_max_weight', 3.0),
    ) if w.get('lambda_hier_child', 0.0) > 0 else outputs['logits_final'].new_tensor(0.0)
    loss_dict['edge'] = edge_bce_loss(outputs['edge_weights'], labels, graph) if model_cfg.get('use_edge', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['concept'] = concept_mse_loss(outputs['concept_scores'], labels, graph) if model_cfg.get('use_concept', True) else outputs['logits_final'].new_tensor(0.0)
    enabled = ['logits_direct']
    if model_cfg.get('use_proto', True): enabled.append('logits_proto')
    if model_cfg.get('use_edge', True): enabled.append('logits_edge')
    if model_cfg.get('use_concept', True): enabled.append('logits_concept')
    loss_dict['consistency'] = branch_consistency_loss(outputs, enabled)
    loss_dict['radius'] = radius_hierarchy_loss(outputs['prototypes'], graph, margin=w.get('radius_margin', 0.05)) if model_cfg.get('use_proto', True) else outputs['logits_final'].new_tensor(0.0)
    if model_cfg.get('use_proto', True) and hp_proto_nce_weight > 0:
        if w.get('hp_proto_mode', 'class') == 'hierarchy':
            loss_dict['hp_proto_nce'] = hyperbolic_hierarchy_prototype_nce_loss(
                outputs,
                labels,
                graph,
                temperature=w.get('hp_temperature', 0.2),
                include_root=w.get('hp_include_root_ancestor', False),
                leaf_weight=w.get('hp_leaf_positive_weight', 1.0),
                ancestor_weight=w.get('hp_ancestor_positive_weight', 0.3),
            )
        else:
            loss_dict['hp_proto_nce'] = hyperbolic_prototype_nce_loss(
                outputs,
                labels,
                graph,
                temperature=w.get('hp_temperature', 0.2),
            )
    else:
        loss_dict['hp_proto_nce'] = outputs['logits_final'].new_tensor(0.0)
    if model_cfg.get('use_proto', True) and hp_supcon_weight > 0:
        loss_dict['hp_supcon'] = hyperbolic_supcon_loss(
            outputs,
            labels,
            temperature=w.get('hp_temperature', 0.2),
        )
    else:
        loss_dict['hp_supcon'] = outputs['logits_final'].new_tensor(0.0)
    if model_cfg.get('use_proto', True) and hp_margin_weight > 0:
        if w.get('hp_margin_mode', 'manual') == 'hierarchy':
            loss_dict['hp_margin'] = hyperbolic_hierarchy_margin_loss(
                outputs,
                labels,
                graph,
                base_margin=w.get('hp_hierarchy_margin_base', w.get('hp_margin', 0.2)),
                max_margin=w.get('hp_hierarchy_margin_max', 0.6),
                neutral_boost=w.get('hp_neutral_margin_boost', 0.2),
                branch_boost=w.get('hp_branch_margin_boost', 0.1),
                same_branch_scale=w.get('hp_same_branch_margin_scale', 0.5),
                prototype_weight=w.get('hp_proto_margin_weight', 0.5),
                pair_weight_base=w.get('hp_pair_weight_base', 1.0),
                neutral_pair_weight=w.get('hp_neutral_pair_weight'),
                neutral_positive_pair_weight=w.get('hp_neutral_positive_pair_weight'),
                neutral_negative_pair_weight=w.get('hp_neutral_negative_pair_weight'),
                cross_branch_pair_weight=w.get('hp_cross_branch_pair_weight'),
                positive_negative_pair_weight=w.get('hp_positive_negative_pair_weight'),
                same_branch_pair_weight=w.get('hp_same_branch_pair_weight'),
                same_negative_pair_weight=w.get('hp_same_negative_pair_weight'),
                pair_reduction=w.get('hp_margin_pair_reduction', 'sample'),
            )
        else:
            loss_dict['hp_margin'] = hyperbolic_class_margin_loss(
                outputs,
                labels,
                graph,
                class_pairs=w.get('hp_margin_pairs', [(0, graph['num_classes'] - 1)]),
                margin=w.get('hp_margin', 0.4),
                prototype_weight=w.get('hp_proto_margin_weight', 0.5),
            )
    else:
        loss_dict['hp_margin'] = outputs['logits_final'].new_tensor(0.0)
    if model_cfg.get('use_proto', True) and subject_ids is not None and hp_subject_centroid_weight > 0:
        loss_dict['hp_subject_centroid'] = hyperbolic_subject_centroid_loss(
            outputs,
            labels,
            subject_ids,
            graph,
            base_margin=w.get('hp_subject_hierarchy_margin_base', 0.2),
            max_margin=w.get('hp_subject_hierarchy_margin_max', 0.6),
            neutral_boost=w.get('hp_subject_neutral_margin_boost', 0.2),
            branch_boost=w.get('hp_subject_branch_margin_boost', 0.1),
            same_branch_scale=w.get('hp_subject_same_branch_margin_scale', 0.5),
            pull_weight=w.get('hp_subject_pull_weight', 1.0),
            margin_weight=w.get('hp_subject_margin_weight', 0.5),
            compactness_weight=w.get('hp_subject_compactness_weight', 0.1),
            pair_weight_base=w.get('hp_subject_pair_weight_base', 1.0),
            neutral_positive_pair_weight=w.get('hp_subject_neutral_positive_pair_weight'),
            neutral_negative_pair_weight=w.get('hp_subject_neutral_negative_pair_weight'),
            positive_negative_pair_weight=w.get('hp_subject_positive_negative_pair_weight'),
            cross_branch_pair_weight=w.get('hp_subject_cross_branch_pair_weight'),
            same_branch_pair_weight=w.get('hp_subject_same_branch_pair_weight'),
            same_negative_pair_weight=w.get('hp_subject_same_negative_pair_weight'),
        )
    else:
        loss_dict['hp_subject_centroid'] = outputs['logits_final'].new_tensor(0.0)
    if model_cfg.get('use_domain_adversarial', False) and subject_ids is not None and 'domain_logits' in outputs:
        loss_dict['domain'] = ce_loss(outputs['domain_logits'], subject_ids)
    else:
        loss_dict['domain'] = outputs['logits_final'].new_tensor(0.0)
    loss_dict['hp_proto_nce_weight'] = outputs['logits_final'].new_tensor(hp_proto_nce_weight)
    loss_dict['hp_supcon_weight'] = outputs['logits_final'].new_tensor(hp_supcon_weight)
    loss_dict['hp_margin_weight'] = outputs['logits_final'].new_tensor(hp_margin_weight)
    loss_dict['hp_subject_centroid_weight'] = outputs['logits_final'].new_tensor(hp_subject_centroid_weight)
    loss_dict['domain_weight'] = outputs['logits_final'].new_tensor(domain_weight)

    total = (
        loss_dict['ce_final']
        + w.get('lambda_direct', 0.3) * loss_dict['ce_direct']
        + w.get('lambda_proto', 0.5) * loss_dict['ce_proto']
        + w.get('lambda_path', 0.0) * loss_dict['ce_path']
        + w.get('lambda_hier_root', 0.0) * loss_dict['hier_root']
        + w.get('lambda_hier_child', 0.0) * loss_dict['hier_child']
        + w.get('lambda_edge', 0.5) * loss_dict['edge']
        + w.get('lambda_concept', 0.1) * loss_dict['concept']
        + w.get('lambda_consistency', 0.1) * loss_dict['consistency']
        + w.get('lambda_radius', 0.01) * loss_dict['radius']
        + domain_weight * loss_dict['domain']
        + hp_proto_nce_weight * loss_dict['hp_proto_nce']
        + hp_supcon_weight * loss_dict['hp_supcon']
        + hp_margin_weight * loss_dict['hp_margin']
        + hp_subject_centroid_weight * loss_dict['hp_subject_centroid']
    )
    loss_dict['total'] = total
    return total, loss_dict
