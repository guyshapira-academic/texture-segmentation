import numpy as np
import scipy as sp
import skimage as ski
import imageio.v3 as imageio


def fetch_brodatz(texture_id: int, imsize: int = 300):
    image_url = f"https://www.ux.uis.no/~tranden/brodatz/D{texture_id}.gif"
    print(image_url)
    image = imageio.imread(image_url)
    image = image[:, 0:imsize, 0:imsize].astype(float) / 255
    return image

def brodatz_test_image(bg_id: int=77, texture_id: int=17, imsize: int = 300):
    bg_image = fetch_brodatz(bg_id, imsize)
    texture_image = fetch_brodatz(texture_id, imsize)
    mask = np.zeros((1, imsize, imsize))
    mask[:, imsize//3:-imsize//3, imsize//3:-imsize//3] = 1
    mask = sp.ndimage.gaussian_filter(mask, sigma=(0, 5, 5))

    texture_image = texture_image - texture_image.mean() + bg_image.mean()
    texture_image = texture_image * bg_image.std() / texture_image.std()
    texture_image = np.clip(texture_image, 0, 1)

    image = (1 - mask) * bg_image + mask * texture_image
    image -= image.min()
    image /= image.max()
    # image = np.random.normal(image, 0.1)
    image = np.clip(image, 0, 1)
    image = image[0]
    return image


def sine_wave_test_image(imsize: int = 300):
    yy, xx = np.mgrid[:imsize, :imsize]
    bg = np.sin(xx / 2)
    phi = np.pi / 2
    inner = np.sin((xx * np.cos(phi) + yy * np.sin(phi)) / 2)

    image = bg
    image[imsize//3:-imsize//3, imsize//3:-imsize//3] = inner[imsize//3:-imsize//3, imsize//3:-imsize//3]
    image -= image.min()
    image /= image.max()

    return image


def disk_on_gradient_test_image(imsize: int = 300):
    yy, xx = np.mgrid[:imsize, :imsize]
    bg = np.sqrt(xx ** 2 + yy ** 2)
    bg = bg.astype(float)
    bg -= bg.min()
    bg /= bg.max()

    image = bg

    rr, cc = ski.draw.disk((imsize//2, imsize//2), imsize//5)
    image[rr, cc] = 1.0

    image -= image.min()
    image /= image.max()

    return image