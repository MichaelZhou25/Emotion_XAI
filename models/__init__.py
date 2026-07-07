from models.eagle_net import EAGLENet


def build_model(cfg, graph):
    return EAGLENet(cfg, graph)
