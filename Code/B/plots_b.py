import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history, results, best_config_name, save_dir="Code/B/plots"):
    ensure_dir(save_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    #Left: Training Curves
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'orange', label='Val Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Curves: {best_config_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    #Right: Bar Chart
    ax2 = axes[1]
    
    exp_names = list(results.keys())
    val_f1_scores = [results[name]['val_f1'] for name in exp_names]
    
    # Shorter names for x-axis
    short_names = ['No Drop\nNo Aug', 'Drop\nNo Aug', 'No Drop\nAug', 'Drop\nAug']
    
    bars = ax2.bar(short_names, val_f1_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    
    # Add value labels on bars
    for bar, score in zip(bars, val_f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Validation F1-Score', fontsize=12)
    ax2.set_title('Regularization Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(save_dir, "model_b_results.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_all_plots(history, results, best_config_name, save_dir="Code/B/plots"):
    filepath = plot_training_curves(history, results, best_config_name, save_dir)
    print(f"Model B plot saved to: {filepath}\n")