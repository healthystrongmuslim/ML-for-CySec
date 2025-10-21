"""Sanity check script to test visualization functions with synthetic data"""
import numpy as np
from visualization import Visualizer


def main():
    visualizer = Visualizer(save_dir='results')

    # Synthetic confusion matrix
    cm = np.array([[50, 5], [3, 42]])
    visualizer.plot_confusion_matrix(cm, class_names=['benign', 'malicious'],
                                    title='Confusion Matrix - Synthetic',
                                    filename='sanity_confusion_matrix.png')

    # ROC curve synthetic
    y_true = np.array([0]*50 + [1]*50)
    y_scores = np.concatenate([np.random.rand(50)*0.2, 0.5 + np.random.rand(50)*0.5])
    visualizer.plot_roc_curve(y_true, y_scores, title='ROC - Synthetic', filename='sanity_roc.png')

    # Training history
    visualizer.plot_training_history([1.0/(i+1) for i in range(100)], filename='sanity_loss.png')

    # Robustness
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    accuracies = [0.95, 0.93, 0.9, 0.85]
    visualizer.plot_robustness_analysis(noise_levels, accuracies, filename='sanity_robustness.png')

if __name__ == '__main__':
    main()
