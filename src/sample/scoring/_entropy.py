import numpy as np


def entropy(class_distribution: np.ndarray) -> float:
    """
    TODO
    """
    return -np.vdot(np.log(class_distribution), class_distribution)
