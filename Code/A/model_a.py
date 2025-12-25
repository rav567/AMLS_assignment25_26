from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def build_random_forest(
    n_estimators=100,
    max_depth=None,
    random_state=42):
    """
    Build a Random Forest classifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

def train_model_a(X_train, y_train, params):
    """
    Train a Random Forest model on training data.
    """
    model = build_random_forest(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model_a(model, X, y):
    """
    Evaluate a tarined model on a given dataset.
    """
    y_pred = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
    }

def retrain_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Retrain the selected Random Forest model using combined
    training and validation data.
    """
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    final_model = build_random_forest(**best_params)
    final_model.fit(X_combined, y_combined)

    return final_model