"""
Preprocessing utilities for classical machine learning models.

This module provides feature extraction methods that convert
28x28 BreastMNIST images into fixed-length feature vectors.
"""

import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
from skimage.feature import hog

def flatten_images(images):
    # Convert (N, 28, 28) images into (N, 784) feature vectors
    return images.reshape(images.shape[0], -1)


def apply_pca(features, n_components=50):
    # Reduce the number of input features (e.g., 784 -> 50) using PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features)
    # Return both the transformed features and the fitted PCA object
    return reduced_features, pca


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