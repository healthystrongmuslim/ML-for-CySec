"""
Visualization utilities for model results and data analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import logging
import os
import config
import utils

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
plt.rcParams['figure.dpi'] = config.DPI


class Visualizer:
    """
    Visualization utilities for the intrusion detection system
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir if save_dir else config.PLOTS_DIR
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None,
                             title: str = 'Confusion Matrix', filename: str = None,
                             normalize: bool = False) -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            filename: Filename to save plot
            normalize: Whether to normalize the matrix
        """
        if normalize:
            # Avoid division by zero when a row sums to 0
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1
            cm = cm.astype('float') / row_sums
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        # Prepare tick labels: use provided class_names or default numeric labels
        n = cm.shape[0]
        if class_names is None:
            tick_labels = [str(i) for i in range(n)]
        else:
            tick_labels = class_names

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=tick_labels, yticklabels=tick_labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved confusion matrix to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = 'ROC Curve', filename: str = None) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True binary labels
            y_scores: Predicted probabilities
            title: Plot title
            filename: Filename to save plot
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved ROC curve to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   title: str = 'Precision-Recall Curve',
                                   filename: str = None) -> None:
        """
        Plot precision-recall curve
        
        Args:
            y_true: True binary labels
            y_scores: Predicted probabilities
            title: Plot title
            filename: Filename to save plot
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'AP = {ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(title)
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved precision-recall curve to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_training_history(self, loss_history: List[float],
                            title: str = 'Training Loss',
                            filename: str = None) -> None:
        """
        Plot training loss history
        
        Args:
            loss_history: List of loss values
            title: Plot title
            filename: Filename to save plot
        """
        plt.figure()
        plt.plot(loss_history, lw=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved training history to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_decision_boundary(self, model, X: np.ndarray, y: np.ndarray,
                             feature_indices: Tuple[int, int] = (0, 1),
                             title: str = 'Decision Boundary',
                             filename: str = None) -> None:
        """
        Plot decision boundary (for 2D visualization)
        
        Args:
            model: Trained model with predict method
            X: Features (will use only 2 features specified)
            y: Labels
            feature_indices: Indices of features to plot
            title: Plot title
            filename: Filename to save plot
        """
        # Extract two features
        X_plot = X[:, feature_indices]
        
        # Create mesh
        h = 0.02  # step size
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh (need to extend to full feature space)
        # This is a simplified version - in practice, need to handle all features
        mesh_samples = np.c_[xx.ravel(), yy.ravel()]
        
        # Create full feature vector (fill other features with mean)
        if X.shape[1] > 2:
            full_mesh = np.zeros((mesh_samples.shape[0], X.shape[1]))
            full_mesh[:, feature_indices[0]] = mesh_samples[:, 0]
            full_mesh[:, feature_indices[1]] = mesh_samples[:, 1]
            # Fill other features with mean values
            for i in range(X.shape[1]):
                if i not in feature_indices:
                    full_mesh[:, i] = X[:, i].mean()
            mesh_samples = full_mesh
        
        Z = model.predict(mesh_samples)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.RdYlBu,
                   edgecolors='black', alpha=0.7)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title(title)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved decision boundary to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_feature_importance(self, importance: np.ndarray,
                               feature_names: List[str] = None,
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               filename: str = None) -> None:
        """
        Plot feature importance
        
        Args:
            importance: Feature importance scores
            feature_names: Names of features
            top_n: Number of top features to show
            title: Plot title
            filename: Filename to save plot
        """
        # Get top N features
        indices = np.argsort(importance)[-top_n:][::-1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved feature importance to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_class_distribution(self, y: np.ndarray, class_names: List[str] = None,
                               title: str = 'Class Distribution',
                               filename: str = None) -> None:
        """
        Plot class distribution
        
        Args:
            y: Labels
            class_names: Names of classes
            title: Plot title
            filename: Filename to save plot
        """
        unique, counts = np.unique(y, return_counts=True)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in unique]
        
        plt.figure()
        plt.bar(range(len(unique)), counts)
        plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title(title)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved class distribution to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict: dict,
                              title: str = 'Model Comparison',
                              filename: str = None) -> None:
        """
        Plot comparison of metrics across models
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics as values
            title: Plot title
            filename: Filename to save plot
        """
        models = list(metrics_dict.keys())
        metric_names = list(metrics_dict[models[0]].keys())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots()
        
        for i, model in enumerate(models):
            values = [metrics_dict[model][m] for m in metric_names]
            ax.bar(x + i * width, values, width, label=model)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend()
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved metrics comparison to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_robustness_analysis(self, noise_levels: List[float],
                                accuracies: List[float],
                                title: str = 'Model Robustness to Noise',
                                filename: str = None) -> None:
        """
        Plot model robustness to adversarial noise
        
        Args:
            noise_levels: List of noise levels
            accuracies: List of accuracies at each noise level
            title: Plot title
            filename: Filename to save plot
        """
        plt.figure()
        plt.plot(noise_levels, accuracies, marker='o', lw=2)
        plt.xlabel('Noise Level')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            logger.info(f"Saved robustness analysis to {filepath}")
        
        plt.show()
        plt.close()


def main():
    """Main function for standalone visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--confusion-matrix', action='store_true')
    parser.add_argument('--roc-curve', action='store_true')
    parser.add_argument('--decision-boundary', action='store_true')
    parser.add_argument('--model', type=str, default='logistic',
                       choices=['logistic', 'svm'])
    
    args = parser.parse_args()
    
    visualizer = Visualizer()
    
    # This is a placeholder - in practice, load actual results
    logger.info("Visualization script - load your model results to visualize")


if __name__ == '__main__':
    main()
