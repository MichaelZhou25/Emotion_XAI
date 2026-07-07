from losses.classification_loss import ce_loss
from losses.edge_loss import edge_bce_loss
from losses.concept_loss import concept_mse_loss
from losses.consistency_loss import branch_consistency_loss
from losses.radius_loss import radius_hierarchy_loss


def compute_eagle_loss(outputs, labels, graph, cfg):
    w = cfg.get('loss', {})
    model_cfg = cfg.get('model', {})
    loss_dict = {}

    loss_dict['ce_final'] = ce_loss(outputs['logits_final'], labels)
    loss_dict['ce_direct'] = ce_loss(outputs['logits_direct'], labels) if model_cfg.get('use_direct_head', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['ce_proto'] = ce_loss(outputs['logits_proto'], labels) if model_cfg.get('use_proto', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['edge'] = edge_bce_loss(outputs['edge_weights'], labels, graph) if model_cfg.get('use_edge', True) else outputs['logits_final'].new_tensor(0.0)
    loss_dict['concept'] = concept_mse_loss(outputs['concept_scores'], labels, graph) if model_cfg.get('use_concept', True) else outputs['logits_final'].new_tensor(0.0)
    enabled = ['logits_direct']
    if model_cfg.get('use_proto', True): enabled.append('logits_proto')
    if model_cfg.get('use_edge', True): enabled.append('logits_edge')
    if model_cfg.get('use_concept', True): enabled.append('logits_concept')
    loss_dict['consistency'] = branch_consistency_loss(outputs, enabled)
    loss_dict['radius'] = radius_hierarchy_loss(outputs['prototypes'], graph, margin=w.get('radius_margin', 0.05)) if model_cfg.get('use_proto', True) else outputs['logits_final'].new_tensor(0.0)

    total = (
        loss_dict['ce_final']
        + w.get('lambda_direct', 0.3) * loss_dict['ce_direct']
        + w.get('lambda_proto', 0.5) * loss_dict['ce_proto']
        + w.get('lambda_edge', 0.5) * loss_dict['edge']
        + w.get('lambda_concept', 0.1) * loss_dict['concept']
        + w.get('lambda_consistency', 0.1) * loss_dict['consistency']
        + w.get('lambda_radius', 0.01) * loss_dict['radius']
    )
    loss_dict['total'] = total
    return total, loss_dict
