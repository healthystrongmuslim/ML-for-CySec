"""
Utility functions for the Intrusion Detection System
"""

import numpy as np
import logging
import json
import pickle
from typing import Tuple, Dict, Any, List
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid function
    
    Args:
        z: Input array
        
    Returns:
        Sigmoid of input
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute softmax function
    
    Args:
        z: Input array of shape (n_samples, n_classes)
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return np.mean(y_true == y_pred)


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                                average: str = 'binary') -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'macro', or 'weighted'
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    elif average in ['macro', 'weighted']:
        classes = np.unique(y_true)
        precisions, recalls, f1s = [], [], []
        weights = []
        
        for cls in classes:
            y_true_cls = (y_true == cls).astype(int)
            y_pred_cls = (y_pred == cls).astype(int)
            
            precision, recall, f1 = compute_precision_recall_f1(y_true_cls, y_pred_cls, 'binary')
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            weights.append(np.sum(y_true == cls))
        
        if average == 'macro':
            return np.mean(precisions), np.mean(recalls), np.mean(f1s)
        else:  # weighted
            weights = np.array(weights) / np.sum(weights)
            return (np.sum(np.array(precisions) * weights),
                   np.sum(np.array(recalls) * weights),
                   np.sum(np.array(f1s) * weights))
    
    else:
        raise ValueError(f"Unknown average method: {average}")


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                     random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def gradient_check(f, x: np.ndarray, analytic_grad: np.ndarray, 
                  epsilon: float = 1e-7) -> float:
    """
    Perform gradient checking using numerical gradients
    
    Args:
        f: Function to compute
        x: Input point
        analytic_grad: Analytically computed gradient
        epsilon: Small value for numerical gradient
        
    Returns:
        Relative error between analytic and numeric gradients
    """
    numeric_grad = np.zeros_like(x)
    
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus.flat[i] += epsilon
        x_minus.flat[i] -= epsilon
        
        numeric_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    # Compute relative error
    numerator = np.linalg.norm(analytic_grad - numeric_grad)
    denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(numeric_grad)
    
    relative_error = numerator / denominator if denominator > 0 else 0
    
    return relative_error


def save_model(model: Any, filename: str) -> None:
    """
    Save model to disk
    
    Args:
        model: Model object
        filename: Path to save model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filename}")


def load_model(filename: str) -> Any:
    """
    Load model from disk
    
    Args:
        filename: Path to model file
        
    Returns:
        Loaded model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {filename}")
    return model


def save_metrics(metrics: Dict[str, Any], filename: str) -> None:
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {filename}")


def load_metrics(filename: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file
    
    Args:
        filename: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(filename, 'r') as f:
        metrics = json.load(f)
    logger.info(f"Metrics loaded from {filename}")
    return metrics


def one_hot_encode(y: np.ndarray, n_classes: int = None) -> np.ndarray:
    """
    Convert labels to one-hot encoded format
    
    Args:
        y: Labels of shape (n_samples,)
        n_classes: Number of classes (if None, inferred from y)
        
    Returns:
        One-hot encoded array of shape (n_samples, n_classes)
    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y.astype(int)] = 1
    
    return one_hot


def add_adversarial_noise(X: np.ndarray, epsilon: float, 
                         gradient: np.ndarray = None) -> np.ndarray:
    """
    Add adversarial noise to data (FGSM attack)
    
    Args:
        X: Input data
        epsilon: Perturbation magnitude
        gradient: Gradient of loss w.r.t. input (if None, use random noise)
        
    Returns:
        Perturbed data
    """
    if gradient is None:
        # Random noise
        noise = np.random.randn(*X.shape)
        noise = noise / np.linalg.norm(noise) * epsilon
    else:
        # FGSM: sign of gradient
        noise = epsilon * np.sign(gradient)
    
    return X + noise


def add_gaussian_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to data
    
    Args:
        X: Input data
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy data
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise
