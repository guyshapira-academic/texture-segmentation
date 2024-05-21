from typing import Dict
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from tqdm import trange


def approximate_delta_function(x: NDArray, eps: float = 1.0) -> NDArray:
    """
    Approximate the delta function.
    """
    divisor = eps * np.sqrt(np.pi)
    exponent = -x ** 2 / eps ** 2
    return np.exp(exponent) / divisor



def pullback_metric(features: NDArray) -> NDArray:
    """
    Compute the pullback metric.
    """
    features = features.reshape(-1, features.shape[-2], features.shape[-1])
    features = (features - features.mean()) / features.std()
    grad = np.gradient(features, axis=(-2, -1))
    grad = np.moveaxis(grad, 0, -1)
    grad = np.moveaxis(grad, 0, -2)
    grad_t = np.moveaxis(grad, -2, -1)
    g = (grad_t @ grad * 1 ) + np.expand_dims(np.eye(2), axis=(0, 1))
    return g


def isotropic_metric(features: NDArray) -> NDArray:
    """
    Compute the isotropic metric.
    """
    g = pullback_metric(features)
    g = 1 / ( np.linalg.det(g) + 1e-7)
    return g


def deodesic_active_contours_segment(
    features: NDArray,
    initial_function: NDArray,
    it: int = 100,
    eta: float = 1e-1,
    c: float = 1,
) -> Dict[str, NDArray]:
    """
    Perform texture segmentation using the geodesic active contours model,
    using the level set method.
    """
    features = features.copy()

    # Standardize the features
    features = (features - features.mean()) / features.std()
    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)
    phi = initial_function.copy()

    g = isotropic_metric(features)
    E = g

    phi_logs = []
    step_logs = []
    fx_logs = []
    fy_logs = []
    grad_magnitude_logs = []

    iterator = trange(it)

    for _ in iterator:
        # phi = sp.ndimage.gaussian_filter(phi, (3, 3), order=(0, 0))
        grad_y, grad_x = np.gradient(phi, axis=(-2, -1), edge_order=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        f_x = E * grad_x / (grad_magnitude + 1e-10)
        f_y = E * grad_y / (grad_magnitude + 1e-10)
        div_f = (
            np.gradient(f_x, axis=(-2, -1), edge_order=2)[1]
            + np.gradient(f_y, axis=(-2, -1), edge_order=2)[0]
        )
        dUdt = grad_magnitude * div_f
        dUdt += c * E * grad_magnitude

        iterator.set_description(f"{np.abs(dUdt).max():.4f}; {grad_magnitude.max():.4f}")

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

    return {
        "phi": phi,
        "E": E,
        "phi_logs": phi_logs,
        "step_logs": step_logs,
        "fx_logs": fx_logs,
        "fy_logs": fy_logs,
        "grad_magnitude_logs": grad_magnitude_logs,
    }


def vector_chan_vase(
    features: NDArray,
    initial_function: NDArray,
    it: int = 100,
    eta: float = 1e-1,
    lambda_c: float = 0.5,
    mu: float = 0.1,
    combined_mathod: bool = False,
) -> Dict[str, NDArray]:
    """
    Perform texture segmentation using the vector-generalized Chan-Vese model.
    """
    features = features.copy()

    # Standardize the features
    features = (features - features.mean()) / features.std()

    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)
    phi = initial_function.copy()

    if combined_mathod:
        g = isotropic_metric(features)
        h = g
    else:
        h = np.ones(shape=features.shape[-2:])

    iterator = trange(it)
    for _ in iterator:
        grad_y, grad_x = np.gradient(phi, axis=(-2, -1), edge_order=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        fx = h * grad_x / (grad_magnitude + 1e-10)
        fy = h * grad_y / (grad_magnitude + 1e-10)
        f_div = np.gradient(fx, axis=(-2, -1), edge_order=2)[1] + np.gradient(fy, axis=(-2, -1), edge_order=2)[0]

        c_in = features[:, phi >= 0].mean(axis=1)
        c_out = features[:, phi < 0].mean(axis=1)

        error_in = (features - c_in[:, None, None]) ** 2
        error_out = (features - c_out[:, None, None]) ** 2
        error_term = - ( (1 - lambda_c) * error_in - lambda_c * error_out).mean(axis=0)

        dphidt = eta * (mu * f_div + (1 - mu) * error_term)
        if combined_mathod:
            dphidt += approximate_delta_function(dphidt, eps=3) * dphidt
        phi += dphidt
    
    return {
        "phi": phi,
        "h": h,
    }