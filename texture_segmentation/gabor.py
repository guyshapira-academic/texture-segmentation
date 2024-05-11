from typing import Tuple
import skimage as ski
import numpy as np
from numpy.typing import NDArray


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
    arr = arr / arr.sum()

    return arr


def gaussian_filter_bank_parameters(
    num_angles: int,
    full_circle: bool = True,
    num_scales: int = 4,
    sigma_x0: float = 0.25,
    sigma_y0_scale: float = 1,
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

    sigma_y0 = np.abs(sigma_y0_scale * x0 * np.sin(angle_delta) / 2)

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
    sigma_y0_scale: float = 1,
) -> NDArray:
    """
    Generate a gaussian filter bank, which a set of gabor filters in the frequency domain.
    """
    locs, sigmas, thetas = gaussian_filter_bank_parameters(
        num_angles=num_angles,
        full_circle=full_circle,
        num_scales=num_scales,
        scaling_factor=scaling_factor,
        sigma_x0=sigma_x0,
        sigma_y0_scale=sigma_y0_scale,
    )
    filters = np.zeros((num_scales, num_angles, size, size), dtype=float)

    for scale, (loc, sigma) in enumerate(zip(locs, sigmas)):
        for angle, theta in enumerate(thetas):
            filters[scale, angle, :, :] = gaussian_filter(size, loc, sigma, theta)

    return filters


def plot_gabor_filter_bank_fft_fwhm(
    size: int,
    num_angles: int = 30,
    full_circle: bool = True,
    num_scales: int = 4,
    scaling_factor: float = 2,
    sigma_x0: float = 0.25,
    sigma_y0_scale: float = 1,
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
        sigma_y0_scale=sigma_y0_scale,
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
