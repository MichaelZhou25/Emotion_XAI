from models.eagle_net import EAGLENet
from models.hemi_mv_eagle_net import HemiMVEAGLENet


def build_model(cfg, graph):
    name = cfg.get('model', {}).get('name', 'EAGLE-Net')
    normalized = name.lower().replace('-', '').replace('_', '')
    if normalized == 'hemimveaglenet':
        return HemiMVEAGLENet(cfg, graph)
    return EAGLENet(cfg, graph)
