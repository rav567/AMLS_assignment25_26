import numpy as np
from skimage.feature import hog

def flatten_images(images):
    # Convert (N, 28, 28) images into (N, 784) feature vectors
    return images.reshape(images.shape[0], -1)


def apply_hog(images, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9):
    # Extract HOG (edge/texture) features from each image
    hog_features = []

    for img in images:
        features = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=False
        )
        hog_features.append(features)

    # Convert list of feature vectors into a NumPy array (N, F)
    return np.array(hog_features)