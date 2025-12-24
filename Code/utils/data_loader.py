"""
Load the BreastMNIST dataset

This flaods the dataset from all the .npz files, normalises them and returns
train, validation and test spliuts in a consistent format for both classical
and deep learning models
"""

import numpy as np
from typing import Tuple
from pathlib import Path


def load_breastmnist(data_path: str = "Datasets/breastmnist.npz"):

    # Check that the dataset file exists before attempting to load it
    assert Path(data_path).exists(), "Dataset file not found"

    # Load the .npz file containing all dataset splits
    data = np.load(data_path)

    # Dictionary to store processed data for each split
    splits = {}

    # Process each dataset split in the same way
    for split in ["train", "val", "test"]:

        #NORMALISATION
        # Load image data for the current split and convert to float
        # Pixel values are normalised from [0, 255] to [0, 1]
        X = data[f"{split}_images"].astype(np.float32) / 255.0

        # Load labels for the current split and remove extra dimensions
        y = data[f"{split}_labels"].squeeze()

        # Store the processed images and labels together
        splits[split] = (X, y)

    # Return train, validation, and test data as separate tuples
    return splits["train"], splits["val"], splits["test"]