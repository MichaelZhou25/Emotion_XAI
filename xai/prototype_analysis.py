import numpy as np


def nearest_prototype(proto_distance):
    return np.argmin(proto_distance, axis=1)


def prototype_margin(proto_distance):
    sorted_d = np.sort(proto_distance, axis=1)
    return sorted_d[:,1] - sorted_d[:,0]
