import numpy as np
from numpy.typing import NDArray
import scipy as sp
from tqdm import trange


def pullback_metric(features: NDArray, normalize_grads: bool = True) -> NDArray:
    """
    Compute the pullback metric.
    """
    features = features.reshape(-1, features.shape[-2], features.shape[-1])
    features = (features - features.mean()) / features.std()
    grad = np.gradient(features, axis=(-2, -1))
    # if normalize_grads:
    #     grad -= np.mean(grad, axis=(-2, -1), keepdims=True)
    #     grad /= np.std(grad, axis=(-2, -1), keepdims=True)
    grad = np.moveaxis(grad, 0, -1)
    grad = np.moveaxis(grad, 0, -2)
    grad_t = np.moveaxis(grad, -2, -1)
    g = grad_t @ grad + np.expand_dims(np.eye(2), axis=(0, 1))
    return g


def isotropic_metric(features: NDArray) -> NDArray:
    """
    Compute the isotropic metric.
    """
    g = pullback_metric(features)
    g = 1 / (np.linalg.det(g) ** 2 + 1e-7)
    return g 


def deodesic_active_contours_segment(
    features: NDArray, initial_function: NDArray, it: int = 100, eta: float = 1e-1, c: float = 1
) -> NDArray:
    """
    Perform texture segmentation using the geodesic active contours model, 
    using the level set method.
    """
    features = features.copy()

    # Standardize the features
    features = (features - features.mean()) / features.std()

    # grad_x, grad_y = np.gradient(features, axis=(-2, -1), edge_order=2)
    # grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # grad_magnitude = np.mean(grad_magnitude, axis=(-2, -1), keepdims=True)
    # features /= (grad_magnitude + 1e-7)

    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)
    phi = initial_function.copy()

    g = isotropic_metric(features)
    E = 1 / np.sqrt(g)
    # E = np.expand_dims(E, axis=0)
    print(E.shape)

    phi_logs = []
    step_logs = []
    fx_logs = []
    fy_logs = []
    grad_magnitude_logs = []
    
    iterator = trange(it)

    for _ in iterator:
        phi = sp.ndimage.gaussian_filter(phi, (3, 3), order=(0, 0))
        grad_x, grad_y = np.gradient(phi, axis=(-2, -1), edge_order=2)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        f_x = E * grad_x / (grad_magnitude + 1e-5)
        f_y = E * grad_y / (grad_magnitude + 1e-5)
        div_f = np.gradient(f_x, axis=(-2, -1), edge_order=2)[1] + np.gradient(f_y, axis=(-2, -1), edge_order=2)[0]
        dUdt = grad_magnitude * div_f
        dUdt += c * E * grad_magnitude

        iterator.set_description(f"{np.abs(dUdt).max():.4f}")

        phi += eta * dUdt

        step_logs.append(dUdt)
        phi_logs.append(phi.copy())

        fx_logs.append(f_x)
        fy_logs.append(f_y)
        grad_magnitude_logs.append(grad_magnitude)

    phi_logs = np.array(phi_logs)
    step_logs = np.array(step_logs)
    fx_logs = np.array(fx_logs)
    fy_logs = np.array(fy_logs)
    grad_magnitude_logs = np.array(grad_magnitude_logs)

    return {"phi": phi, "E": E, "phi_logs": phi_logs, "step_logs": step_logs, "fx_logs": fx_logs, "fy_logs": fy_logs, "grad_magnitude_logs": grad_magnitude_logs}
