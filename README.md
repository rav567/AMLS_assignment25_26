# AMLS_assignment25_26

## Overview

This project is part of the **Applied Machine Learning Systems (AMLS)** coursework.
The aim is to benchmark and analyse the performance of two different machine learning approaches on a medical image classification task using the **BreastMNIST** dataset.

The project compares:

* **Model A**: A classical machine learning pipeline using a Random Forest classifier with handcrafted feature extraction.
* **Model B**: A convolutional neural network (CNN) implemented using PyTorch.

Both models are evaluated under different configurations to analyse the effects of:

* Model capacity
* Data augmentation
* Training strategy and reproducibility

The task is **binary classification** of breast ultrasound images into *benign* or *malignant* classes.

---

## Project Structure

```
AMLS_25_26_SN25261710/
│
├── Code/
│   ├── A/
│   │   ├── model_a.py        # Model A (Random Forest) implementation
│   │   ├── plots_a.py        # Plotting utilities for Model A (disabled in main)
│   │   └── plots/            # Output directory for plots (optional)
│   │
│   ├── B/
│   │   ├── model_b.py        # Model B (CNN) implementation
│   │   ├── plots_b.py        # Plotting utilities for Model B (disabled in main)
│   │   └── plots/            # Output directory for plots (optional)
│   │
│   └── utils/
│       ├── data_loader.py    # Dataset loading logic
│       ├── preprocessing.py  # Feature extraction utilities
│       └── augmentation.py   # Image augmentation utilities
│
├── Datasets/                 # Leave EMPTY for submission
│
├── main.py                   # Entry point to run all experiments
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation
└── .gitignore
```

---

## Dataset

* **Dataset name**: BreastMNIST
* **Format**: `breastmnist.npz`
* **Image size**: 28 × 28 (grayscale)

⚠️ **Important**
The `Datasets/` folder will be **empty when submitting** this project.
During assessment, the dataset will be copied into this folder.

Expected structure during execution:

```
Datasets/
└── breastmnist.npz
```

---

## Requirements

* **Python version**: Python 3.8 or newer

### Required packages

Install all dependencies using:

```bash
pip install -r requirements.txt
```

The project uses:

* NumPy
* scikit-learn
* PyTorch
* Matplotlib and Seaborn (for analysis plots)
* SciPy, Pillow, tqdm (supporting dependencies)

---

## How to Run the Project

Once the dataset has been placed into the `Datasets/` folder, run:

```bash
python main.py
```

This will:

1. Load the BreastMNIST dataset
2. Train and evaluate **Model A**
3. Train and evaluate **Model B**
4. Print the final test performance for both models to the terminal

No additional arguments are required.

---

## Output

The program prints the **final test metrics** for each model, including:

* Accuracy
* Precision
* Recall
* F1-score

Intermediate logging (such as per-epoch training output and plot generation) has been **intentionally disabled** to keep execution clean and suitable for assessment.

---

## Notes on Plotting

The project includes plotting utilities for analysing:

* Model capacity
* Feature pipelines
* Overfitting behaviour
* Training dynamics

These plotting functions are **not executed by default** and are commented out in `main.py`.
They were used during development and for generating figures used in the accompanying report, and can be re-enabled if required.

---

## Reproducibility

To ensure reproducibility:

* Fixed random seeds are used for NumPy and PyTorch
* Deterministic behaviour is enforced where supported by the backend

This helps ensure consistent results across runs and environments.

---

## Project Status

This project is complete and submitted as coursework for the AMLS module.
No further development is planned.
