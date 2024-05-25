from typing import Dict, Literal, Union
import enum
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from tqdm import trange

from texture_segmentation import gabor
import skimage as ski

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
    g = (grad_t @ grad) + np.expand_dims(np.eye(2), axis=(0, 1))
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
        grad_y, grad_x = sp.ndimage.gaussian_filter(phi, sigma=3, order=(1, 0)), sp.ndimage.gaussian_filter(phi, sigma=3, order=(0, 1))
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        f_x = E * grad_x / (grad_magnitude + 1e-10)
        f_y = E * grad_y / (grad_magnitude + 1e-10)
        div_f = (
            sp.ndimage.gaussian_filter(f_x, sigma=3, order=(0, 1))
            + sp.ndimage.gaussian_filter(f_y, sigma=3, order=(1, 0))
        )
        dUdt = div_f * grad_magnitude
        dUdt += c * E * grad_magnitude

        # iterator.set_description(f"{np.abs(dUdt).max():.4f}; {grad_magnitude.max():.4f}")

        phi += eta * dUdt
        # phi = sp.ndimage.gaussian_filter(phi, sigma=1)

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
        "edges": E,
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
    combined_mathod: bool = True,
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
            dphidt += approximate_delta_function(dphidt, eps=1) * dphidt
        phi += dphidt
    
    return {
        "phi": phi,
        "edges": h,
    }


class SEG_METHOD(enum.Enum):
    GEODESIC_SNAKES = enum.auto()
    ACTIVE_CONTOURS = enum.auto()

class FEATURE_TYPE(enum.Enum):
    GABOR = enum.auto()
    HIGH_LEVEL = enum.auto()
    IMAGE = enum.auto()


def segment(
    image: NDArray,
    method: SEG_METHOD,
    feature_type: FEATURE_TYPE = FEATURE_TYPE.GABOR,
    initial_function: Union[str, NDArray] = "random",
    it: int = 100,
    eta: float = 1e-1,
    gabor_filters_params: Dict[str, float] = None,
    **kwargs,
) -> Dict[str, NDArray]:
    """
    Perform texture segmentation.
    """
    # Compute the features
    gabor_features, hl_features, names = gabor.gabor_features(image, gabor_filters_params)
    hl_features -= hl_features.mean(axis=(-2, -1), keepdims=True)
    hl_features /= (hl_features.std(axis=(-2, -1), keepdims=True) + 1e-6)
    gabor_features = gabor.features_post_process(gabor_features, sigma=5, diffusion_eta=0.1, diffusion_steps=20)
    hl_features = gabor.features_post_process(hl_features, sigma=3, diffusion_eta=0.1, diffusion_steps=20)

    if feature_type == FEATURE_TYPE.GABOR:
        features = gabor_features
    elif feature_type == FEATURE_TYPE.HIGH_LEVEL:
        features = hl_features
    elif feature_type == FEATURE_TYPE.IMAGE:
        features = image

    imsize = image.shape[-1]
    if isinstance(initial_function, str):
        if initial_function == "disk":
            phi0 = 2 * ski.segmentation.disk_level_set(image_shape=(imsize, imsize), radius=imsize//4).astype(float) - 1
            phi0 = sp.ndimage.gaussian_filter(phi0, sigma=5)
        elif initial_function == "checkers":
            phi0 = 2 * ski.segmentation.checkerboard_level_set((imsize, imsize), 10).astype(float) - 1
            phi0 = sp.ndimage.gaussian_filter(phi0, sigma=2)
        elif initial_function == "random":
            phi0 = np.random.randn(imsize, imsize) * 0.1
            phi0 = sp.ndimage.gaussian_filter(phi0, sigma=5)
        else:
            raise ValueError("Invalid initial function")      
    else:
        phi0 = initial_function 

    if method == SEG_METHOD.GEODESIC_SNAKES:
        res = vector_chan_vase(
            features, phi0, it=it, eta=eta, **kwargs
        )
    elif method == SEG_METHOD.ACTIVE_CONTOURS:
        res = deodesic_active_contours_segment(
            features, phi0, it=it, eta=eta, **kwargs
        )
    else:
        raise ValueError("Invalid segmentation method")
    
    res["features"] = features
    res["image"] = image
    res["phi0"] = phi0
    res["hl_features"] = hl_features
    res["gabor_features"] = gabor_features

    return res