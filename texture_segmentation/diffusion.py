from typing import Tuple
import numpy as np
import scipy as sp
from numpy.typing import NDArray

from tqdm import trange


def metric(features: NDArray):
    features = features.reshape(-1, features.shape[-2], features.shape[-1])
    grad = np.gradient(features, axis=(-2, -1))
    grad = np.moveaxis(grad, 0, -1)
    grad = np.moveaxis(grad, 0, -2)
    grad_t = np.moveaxis(grad, -2, -1)
    g = grad_t @ grad + np.expand_dims(np.eye(2), axis=(0, 1))
    return g


def diffuse_features(features, it=20, eta=1e-1):
    features = features.copy()
    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)
    for _ in trange(it):
        grad_x, grad_y = np.gradient(features, axis=(-2, -1), edge_order=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        fx = grad_x / (1 + grad_magnitude**2)
        fy = grad_y / (1 + grad_magnitude**2)

        f = (
            np.gradient(fx, axis=(-2, -1), edge_order=2)[1]
            + np.gradient(fy, axis=(-2, -1), edge_order=2)[0]
        )

        # f = sp.ndimage.gaussian_filter(
        #     fx, (0, 1, 1), order=(0, 0, 1)
        # ) + sp.ndimage.gaussian_filter(fy, (0, 1, 1), order=(0, 1, 0))

        g = metric(features)
        g = np.sqrt(np.linalg.det(g))
        g = np.expand_dims(g, axis=0)
        f /= g

        features += eta * f
    return features
