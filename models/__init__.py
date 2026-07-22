from models.eagle_net import EAGLENet
from models.hemi_mv_eagle_net import HemiMVEAGLENet
from models.hyperbolic_path_eagle_net import (
    HyperbolicPathEAGLENet,
    NeutralCenteredHyperbolicPathEAGLENet,
)


def build_model(cfg, graph):
    name = cfg.get('model', {}).get('name', 'EAGLE-Net')
    normalized = name.lower().replace('-', '').replace('_', '')
    if normalized == 'neutralcenteredhyperbolicpatheaglenet':
        return NeutralCenteredHyperbolicPathEAGLENet(cfg, graph)
    if normalized == 'hyperbolicpatheaglenet':
        return HyperbolicPathEAGLENet(cfg, graph)
    if normalized == 'hemimveaglenet':
        return HemiMVEAGLENet(cfg, graph)
    return EAGLENet(cfg, graph)
