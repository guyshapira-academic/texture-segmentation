from typing import Any, Dict, Optional, Tuple
import skimage as ski
import numpy as np
import scipy as sp
from numpy.typing import NDArray

from texture_segmentation import diffusion

_default_filter_bank_params = {
    "num_angles": 20,
    "full_circle": False,
    "scaling_factor": 0.4,
    "num_scales": 4,
    "sigma_x0": 0.17,
    "sigma_y0": 0.07,
}


def gaussian_filter(
    size: int, loc: Tuple[float, float], sigma: Tuple[float, float], theta: float
) -> NDArray:
    """
    Generate a gaussian function over a square array.
    The gaussian is parametrized w.r.t. the region [-1, 1]x[-1, 1].
    """
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]

    # Scale to [-1, 1]x[-1, 1]
    x = x / np.abs(x).max()
    y = y / np.abs(y).max()

    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    exponent_ = -0.5 * (
        (x_rot - loc[0]) ** 2 / sigma[0] ** 2 + (y_rot - loc[1]) ** 2 / sigma[1] ** 2
    )
    arr = np.exp(exponent_)

    # Normalize
    arr = arr / np.linalg.norm(arr)

    return arr


def gaussian_filter_bank_parameters(
    num_angles: int,
    full_circle: bool = True,
    num_scales: int = 4,
    sigma_x0: float = 0.25,
    sigma_y0: float = 0.1,
    scaling_factor: float = 0.2,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate gaussian parameters for a filter bank.
    """
    fwhm_constant = np.sqrt(2 * np.log(2))
    x0 = 1 / (1 + fwhm_constant * sigma_x0)

    if full_circle:
        angle_delta = 2 * np.pi / num_angles
    else:
        angle_delta = np.pi / num_angles

    sigma_y0 = sigma_y0

    sigmas = [
        (sigma_x0 * (scaling_factor ** (k / 2)), sigma_y0 * (scaling_factor ** (k / 2)))
        for k in range(0, num_scales)
    ]

    locs = [(x0, 0)]
    for k in range(1, num_scales):
        next_ = (
            locs[-1][0]
            - sigmas[k - 1][0] * fwhm_constant
            - sigmas[k][0] * fwhm_constant
        )
        locs.append((next_, 0))
    thetas = [angle_delta * k for k in range(num_angles)]

    return locs, sigmas, thetas


def gabor_filter_bank_fft(
    size: int,
    num_angles: int = 30,
    full_circle: bool = True,
    num_scales: int = 4,
    scaling_factor: float = 0.2,
    sigma_x0: float = 0.25,
    sigma_y0: float = 0.1,
) -> Tuple[NDArray, NDArray]:
    """
    Generate a gaussian filter bank, which a set of gabor filters in the frequency domain.
    """
    locs, sigmas, thetas = gaussian_filter_bank_parameters(
        num_angles=num_angles,
        full_circle=full_circle,
        num_scales=num_scales,
        scaling_factor=scaling_factor,
        sigma_x0=sigma_x0,
        sigma_y0=sigma_y0,
    )
    filters = np.zeros((num_scales, num_angles, size, size), dtype=float)

    for scale, (loc, sigma) in enumerate(zip(locs, sigmas)):
        for angle, theta in enumerate(thetas):
            filters[scale, angle, :, :] = gaussian_filter(size, loc, sigma, theta)

    return filters, (locs, sigmas, thetas)


def plot_gabor_filter_bank_fft_fwhm(
    size: int,
    num_angles: int = 30,
    full_circle: bool = True,
    num_scales: int = 4,
    scaling_factor: float = 2,
    sigma_x0: float = 0.25,
    sigma_y0: float = 0.1,
) -> NDArray:
    """
    Plot ellipses where the FWHM of the gabor filters are.
    """
    fwhm_constant = np.sqrt(2 * np.log(2))
    image = np.zeros((size, size), dtype=float)
    locs, sigmas, thetas = gaussian_filter_bank_parameters(
        num_angles=num_angles,
        full_circle=full_circle,
        num_scales=num_scales,
        scaling_factor=scaling_factor,
        sigma_x0=sigma_x0,
        sigma_y0=sigma_y0,
    )

    for _, (loc, sigma) in enumerate(zip(locs, sigmas)):
        for _, theta in enumerate(thetas):
            rx = int(sigma[1] * size * fwhm_constant / 2)
            ry = int(sigma[0] * size * fwhm_constant / 2)
            theta = -theta
            loc_rot = (
                np.cos(theta) * loc[0] - np.sin(theta) * loc[1],
                np.sin(theta) * loc[0] + np.cos(theta) * loc[1],
            )
            x_center = int((loc_rot[1] + 1) * size // 2)
            y_center = int((loc_rot[0] + 1) * size // 2)

            rr, cc = ski.draw.ellipse_perimeter(
                x_center, y_center, rx, ry, theta, (size, size)
            )
            image[rr, cc] = 1

    return image


def gabor_features_raw(
    image: NDArray, gabor_filters_params: Optional[Dict[str, Any]] = None
) -> NDArray:
    global _default_gabor_params

    assert image.ndim == 2 or image.ndim == 3
    init_dim = image.ndim
    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    assert image.shape[-2] == image.shape[-1]
    size = image.shape[-1]

    if gabor_filters_params is None:
        gabor_filters_params = _default_gabor_params

    gabor_filters_fft, _ = gabor_filter_bank_fft(size=size, **gabor_filters_params)

    c, h, w = image.shape
    num_scales, num_angles, _, _ = gabor_filters_fft.shape
    image = image.reshape(c, 1, 1, h, w)
    gabor_filters_fft = gabor_filters_fft.reshape(1, num_scales, num_angles, h, w)
    gabor_filters_fft = np.fft.ifftshift(gabor_filters_fft, axes=(-2, -1))
    image_fft = np.fft.fft2(image, axes=(-2, -1), norm="ortho")
    gabor_features_fft = gabor_filters_fft * image_fft
    gabor_features = np.fft.ifft2(gabor_features_fft, axes=(-2, -1), norm="ortho")

    if init_dim == 2:
        gabor_features = gabor_features[0]

    return gabor_features


def gabor_features(
    image: NDArray, gabor_filters_params: Optional[Dict[str, Any]] = None
) -> NDArray:
    global _default_gabor_params
    if gabor_filters_params is None:
        gabor_filters_params = _default_gabor_params
    bank_params = gaussian_filter_bank_parameters(**gabor_filters_params)
    locs = bank_params[0]
    locs = np.array(locs)[:, 0]
    raw_features = gabor_features_raw(image, gabor_filters_params)
    gabor_response = np.abs(raw_features)

    n, m = gabor_response.shape[:2]
    gabor_response = gabor_response.reshape(n * m, *gabor_response.shape[2:])
    max_idx = np.argmax(gabor_response, axis=0)
    # gabor_resoponse = gabor_resoponse.reshape(n, m, *gabor_resoponse.shape[1:])
    max_freq_idx, max_angle_idx = np.unravel_index(max_idx, (n, m))
    xx, yy = np.mgrid[0:gabor_response.shape[-2], 0:gabor_response.shape[-1]]
    max_complex_gabor_response = raw_features[max_freq_idx, max_angle_idx, xx, yy]
    max_real_gabor_response = max_complex_gabor_response.real
    max_imag_gabor_response = max_complex_gabor_response.imag
    max_angle = max_angle_idx * np.pi / gabor_filters_params['num_angles']
    max_angle_sin = np.sin(max_angle)
    max_angle_cos = np.cos(max_angle)
    init_shape = max_freq_idx.shape
    max_freq = locs[max_freq_idx.flatten()].reshape(init_shape)
       

    hl_features = np.stack([max_angle_cos, max_angle_sin, max_freq, max_real_gabor_response, max_imag_gabor_response], axis=0)
    return gabor_response, hl_features, ("angle_cos", "angle_sin", "freq", "max_real", "max_imag")

def features_post_process(features: NDArray, sigma: float = 5, diffusion_eta: float = 0.1, diffusion_steps: int = 50) -> NDArray:
    features = features.copy()
    features = sp.ndimage.gaussian_filter(features, sigma=(0, sigma, sigma))
    features = diffusion.diffuse_features(features, it=diffusion_steps, eta=diffusion_eta)
    
    return features