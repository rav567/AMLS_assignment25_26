from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from Code.utils.preprocessing import flatten_images, apply_hog
from Code.utils.augmentation import augment_image


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

def run_model_a(train_data, val_data, test_data):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. EXTRACT FEATURES
    X_train_raw = flatten_images(X_train)
    X_val_raw = flatten_images(X_val)
    X_test_raw = flatten_images(X_test)
    
    X_train_hog = apply_hog(X_train)
    X_val_hog = apply_hog(X_val)
    X_test_hog = apply_hog(X_test)
    
    # 2. RUN EXPERIMENTS
    experiments = [
        ("Raw", "Low",  X_train_raw, X_val_raw, {"n_estimators": 50,  "max_depth": 5,  "random_state": 42}),
        ("Raw", "High", X_train_raw, X_val_raw, {"n_estimators": 200, "max_depth": None, "random_state": 42}),
        ("HOG", "Low",  X_train_hog, X_val_hog, {"n_estimators": 50,  "max_depth": 5,  "random_state": 42}),
        ("HOG", "High", X_train_hog, X_val_hog, {"n_estimators": 200, "max_depth": None, "random_state": 42})
    ]
    
    results = {}
    
    for feature, capacity, Xtr, Xval, params in experiments:
        model = train_model_a(Xtr, y_train, params)
        train_metrics = evaluate_model_a(model, Xtr, y_train)
        val_metrics = evaluate_model_a(model, Xval, y_val)
        
        results[(feature, capacity)] = {
            "model": model,
            "train_f1": train_metrics["F1"],
            "val_f1": val_metrics["F1"],
            "val_metrics": val_metrics
        }
    
    # 3. TEST AUGMENTATION
    best_key = max(results.keys(), key=lambda k: results[k]["val_f1"])
    best_feature, best_capacity = best_key
    
    X_train_aug = np.concatenate([X_train, X_train, X_train], axis=0)
    y_train_aug = np.concatenate([y_train, y_train, y_train], axis=0)
    
    for i in range(len(X_train), len(X_train_aug)):
        X_train_aug[i] = augment_image(X_train_aug[i])
    
    if best_feature == "Raw":
        X_train_aug_features = flatten_images(X_train_aug)
        Xval_features = X_val_raw
    else:
        X_train_aug_features = apply_hog(X_train_aug)
        Xval_features = X_val_hog
    
    best_params = experiments[[e[:2] for e in experiments].index(best_key)][4]
    
    model_aug = train_model_a(X_train_aug_features, y_train_aug, best_params)
    val_metrics_aug = evaluate_model_a(model_aug, Xval_features, y_val)
    
    if val_metrics_aug['F1'] > results[best_key]['val_f1']:
        results[(f"{best_feature}+Aug", best_capacity)] = {
            "model": model_aug,
            "val_f1": val_metrics_aug['F1'],
            "val_metrics": val_metrics_aug
        }
        best_key = (f"{best_feature}+Aug", best_capacity)
    
    # 4. DATA EFFICIENCY TEST
    ratios = [0.1, 0.3, 0.6, 1.0]
    
    if best_feature.startswith("Raw"):
        Xtr, Xval = X_train_raw, X_val_raw
    else:
        Xtr, Xval = X_train_hog, X_val_hog
    
    for ratio in ratios:
        n = int(ratio * len(Xtr))
        model = train_model_a(Xtr[:n], y_train[:n], best_params)
        f1 = evaluate_model_a(model, Xval, y_val)["F1"]
    
    # 5. FINAL TEST EVALUATION
    final_feature = best_feature.replace("+Aug", "")
    
    if final_feature == "Raw":
        Xtr_final, Xval_final, Xtest_final = X_train_raw, X_val_raw, X_test_raw
    else:
        Xtr_final, Xval_final, Xtest_final = X_train_hog, X_val_hog, X_test_hog
    
    final_model = retrain_final_model(Xtr_final, y_train, Xval_final, y_val, best_params)
    test_metrics = evaluate_model_a(final_model, Xtest_final, y_test)
    
    # Formatting
    best_config = f"{best_feature} + {best_capacity}"
    
    return test_metrics, best_config, results