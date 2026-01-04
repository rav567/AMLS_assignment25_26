import numpy as np
from scipy.ndimage import rotate

def random_horizontal_flip(image, p: float = 0.5):
    if np.random.rand() < p:
        image = np.fliplr(image)
    return image

def random_rotation(image, max_angle: float = 15.0, p: float = 0.5):
    if np.random.rand() < p:
        angle = np.random.uniform(-max_angle, max_angle)
        image = rotate(image, angle, reshape=False, mode="nearest")
    return image

def random_brightness(image, max_delta: float = 0.1, p: float = 0.5):
    if np.random.rand() < p:
        delta = np.random.uniform(-max_delta, max_delta)
        image = image + delta
    return image

def random_contrast(image, max_factor: float = 0.1, p: float = 0.5):
    if np.random.rand() < p:
        factor = 1.0 + np.random.uniform(-max_factor, max_factor)
        mean = np.mean(image)
        image = (image - mean) * factor + mean
    return image

def augment_image(image):
    image = random_horizontal_flip(image)
    image = random_rotation(image)
    image = random_brightness(image)
    image = random_contrast(image)
    image = np.clip(image, 0.0, 1.0)

    return image