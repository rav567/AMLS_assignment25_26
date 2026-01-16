import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_feature_capacity_heatmap(results, save_dir="Code/A/plots"):
    ensure_dir(save_dir)
    
    # Extract unique features and capacities (excluding augmented versions)
    base_results = {k: v for k, v in results.items() if "+Aug" not in k[0]}
    
    features = sorted(set(k[0] for k in base_results.keys()))
    capacities = sorted(set(k[1] for k in base_results.keys()), 
                       key=lambda x: 0 if x == "Low" else 1)
    
    data = np.array([
        [base_results[(feature, cap)]["val_f1"] for cap in capacities]
        for feature in features])
    
    # Create heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=capacities,
        yticklabels=features,
        cbar_kws={"label": "Validation F1-score"}
    )
    
    plt.xlabel("Model Capacity", fontsize=12)
    plt.ylabel("Feature Pipeline", fontsize=12)
    plt.title("Model A: Feature Pipeline × Capacity Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(save_dir, "feature_capacity_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overfitting_gap(results, save_dir="Code/A/plots"):
    ensure_dir(save_dir)
    
    # Extract base results (no augmentation)
    base_results = {k: v for k, v in results.items() if "+Aug" not in k[0]}
    
    # Group by feature
    features = sorted(set(k[0] for k in base_results.keys()))
    capacities = ["Low", "High"]
    
    plt.figure(figsize=(8, 5))
    
    for feature in features:
        x = []
        y = []
        for capacity in capacities:
            key = (feature, capacity)
            if key in base_results:
                gap = base_results[key]["train_f1"] - base_results[key]["val_f1"]
                x.append(capacity)
                y.append(gap)
        
        plt.plot(x, y, marker="o", markersize=8, linewidth=2, label=feature)

    plt.axhline(0, linestyle="--", color="grey", alpha=0.6, label="Perfect generalisation")
    
    plt.xlabel("Model Capacity", fontsize=12)
    plt.ylabel("Generalisation Gap (Train F1 − Val F1)", fontsize=12)
    plt.title("Model A: Overfitting Analysis", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(save_dir, "overfitting_gap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(results, save_dir="Code/A/plots"):
    plot_feature_capacity_heatmap(results, save_dir)
    plot_overfitting_gap(results, save_dir)