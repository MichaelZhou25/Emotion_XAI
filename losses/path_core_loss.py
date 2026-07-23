from losses.classification_loss import ce_loss
from losses.edge_loss import edge_bce_loss
from losses.hyperbolic_contrastive_loss import hyperbolic_hierarchy_prototype_nce_loss


def compute_path_core4_loss(outputs, labels, graph, cfg):
    loss_cfg = cfg['loss']
    label_smoothing = cfg.get('train', {}).get('label_smoothing', 0.0)

    losses = {
        'path': ce_loss(outputs['logits_edge'], labels, label_smoothing),
        'edge': edge_bce_loss(
            outputs['edge_weights'],
            labels,
            graph,
            edge_scores=outputs['edge_scores'],
            positive_weight=loss_cfg.get('edge_positive_weight', 1.0),
        ),
        'hpcl': hyperbolic_hierarchy_prototype_nce_loss(
            outputs,
            labels,
            graph,
            temperature=loss_cfg.get('hpcl_temperature', 0.2),
            include_root=loss_cfg.get('hpcl_include_root', False),
            leaf_weight=loss_cfg.get('hpcl_leaf_weight', 1.0),
            ancestor_weight=loss_cfg.get('hpcl_ancestor_weight', 0.3),
        ),
        'direct': ce_loss(outputs['logits_direct'], labels, label_smoothing),
    }
    total = (
        float(loss_cfg.get('lambda_path', 1.0)) * losses['path']
        + float(loss_cfg.get('lambda_edge', 0.1)) * losses['edge']
        + float(loss_cfg.get('lambda_hpcl', 0.02)) * losses['hpcl']
        + float(loss_cfg.get('lambda_direct', 0.3)) * losses['direct']
    )
    losses['total'] = total
    return total, losses
