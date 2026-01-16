"""
This is the beginning of the AMLS BreastMNIST coursework
"""

from Code.utils.data_loader import load_breastmnist
from Code.A.model_a import run_model_a
from Code.B.model_b import run_model_b

# Plotting utilities (disabled for final submission)
# from Code.A.plots_a import generate_all_plots as generate_plots_a
# from Code.B.plots_b import generate_all_plots as generate_plots_b

def main():
    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_breastmnist()
    
    # Run all Model A experiments and get final test results
    test_metrics, best_config, results_a = run_model_a(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test))
    
    # Plot generation disabled for submission
    #generate_plots_a(results_a, save_dir="Code/A/plots")

    test_metrics_b, best_config_b, results_b, history_b = run_model_b(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test))

    #Plot generation disabled for submission
    #generate_plots_b(history_b, results_b, best_config_b, save_dir="Code/B/plots")
    
    # Print final results
    print("Model A Results")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.3f}")
    print(f"  Precision: {test_metrics['Precision']:.3f}")
    print(f"  Recall:    {test_metrics['Recall']:.3f}")
    print(f"  F1-score:  {test_metrics['F1']:.3f}")

    print("\nModel B Final Results:")
    print(f"  Accuracy:  {test_metrics_b['Accuracy']:.3f}")
    print(f"  Precision: {test_metrics_b['Precision']:.3f}")
    print(f"  Recall:    {test_metrics_b['Recall']:.3f}")
    print(f"  F1-score:  {test_metrics_b['F1']:.3f}")

if __name__ == "__main__":
    main()