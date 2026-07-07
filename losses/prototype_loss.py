from losses.classification_loss import ce_loss


def prototype_ce_loss(logits_proto, labels):
    return ce_loss(logits_proto, labels)
