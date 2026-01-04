"""
This is the beginning of the AMLS BreastMNIST coursework
"""

from Code.utils.data_loader import load_breastmnist
from Code.A.plots_a import generate_all_plots
from Code.A.model_a import run_model_a
#from Code.B.model_b import run_model_b

def main():
    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist()
    
    # Run all Model A experiments and get final test results
    test_metrics, best_config, results = run_model_a(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test))
    
    generate_all_plots(results, save_dir="Code/A/plots")
    
    # Print final results
    print("Model A Results")
    print(f"Best Configuration: {best_config}")
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.3f}")
    print(f"  Precision: {test_metrics['Precision']:.3f}")
    print(f"  Recall:    {test_metrics['Recall']:.3f}")
    print(f"  F1-score:  {test_metrics['F1']:.3f}")

if __name__ == "__main__":
    main()