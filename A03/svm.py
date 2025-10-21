"""
Support Vector Machine (SVM) implementation from scratch
Supports multiple kernel functions
"""

import numpy as np
from typing import Callable, Optional
import logging
import utils
import config

logger = logging.getLogger(__name__)


class SVM:
    """
    Support Vector Machine classifier implemented from scratch
    
    Supports:
    - Binary and multi-class classification (one-vs-rest)
    - Multiple kernels: linear, polynomial, RBF
    - Gradient descent optimization with hinge loss
    - Regularization
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', gamma: float = 'scale',
                 degree: int = 3, max_iterations: int = 1000, tolerance: float = 1e-3,
                 learning_rate: float = 0.001, verbose: bool = True):
        """
        Initialize SVM model
        
        Args:
            C: Regularization parameter (inverse of regularization strength)
            kernel: Kernel type ('linear', 'polynomial', 'rbf')
            gamma: Kernel coefficient for rbf and polynomial ('scale', 'auto', or float)
            degree: Degree for polynomial kernel
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print training progress
        """
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.is_binary = None
        self.models = []  # For multi-class (one-vs-rest)
        self.loss_history = []
        self.gamma_value = None
        
    def _compute_gamma(self, X: np.ndarray) -> float:
        """Compute gamma value for kernel"""
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        else:
            return float(self.gamma)
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel: K(x1, x2) = x1^T * x2"""
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Polynomial kernel: K(x1, x2) = (gamma * x1^T * x2 + 1)^degree"""
        return (self.gamma_value * np.dot(X1, X2.T) + 1) ** self.degree
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
        # Compute squared Euclidean distances
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma_value * distances)
    
    def _get_kernel_function(self) -> Callable:
        """Get the kernel function based on kernel name"""
        if self.kernel_name == 'linear':
            return self._linear_kernel
        elif self.kernel_name == 'polynomial':
            return self._polynomial_kernel
        elif self.kernel_name == 'rbf':
            return self._rbf_kernel
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def _compute_hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute hinge loss with L2 regularization
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels (must be -1 or 1)
            
        Returns:
            Loss value
        """
        n_samples = X.shape[0]
        
        # Compute decision function
        decision = np.dot(X, self.weights) + self.bias
        
        # Hinge loss: max(0, 1 - y * decision)
        hinge_loss = np.maximum(0, 1 - y * decision)
        
        # Total loss with regularization
        loss = np.mean(hinge_loss) + (1 / (2 * self.C)) * np.sum(self.weights ** 2)
        
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Compute gradients for hinge loss
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels (must be -1 or 1)
            
        Returns:
            Gradients for weights and bias
        """
        n_samples = X.shape[0]
        
        # Compute decision function
        decision = np.dot(X, self.weights) + self.bias
        
        # Compute margin
        margin = y * decision
        
        # Gradient of hinge loss
        # If margin >= 1: gradient is 0
        # If margin < 1: gradient is -y
        dw = np.zeros_like(self.weights)
        db = 0
        
        for i in range(n_samples):
            if margin[i] < 1:
                dw -= y[i] * X[i]
                db -= y[i]
        
        # Average and add regularization
        dw = (dw / n_samples) + (1 / self.C) * self.weights
        db = db / n_samples
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train the SVM model
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels
            
        Returns:
            self
        """
        self.n_classes = len(np.unique(y))
        self.is_binary = (self.n_classes == 2)
        
        # Compute gamma value
        self.gamma_value = self._compute_gamma(X)
        
        if self.is_binary:
            # Binary classification
            # Convert labels to -1 and 1
            y_binary = np.where(y == 0, -1, 1)
            self._fit_binary(X, y_binary)
        else:
            # Multi-class classification (one-vs-rest)
            self.models = []
            for class_idx in range(self.n_classes):
                if self.verbose:
                    logger.info(f"Training binary classifier for class {class_idx}")
                
                # Create binary labels
                y_binary = np.where(y == class_idx, 1, -1)
                
                # Create and train binary SVM
                binary_svm = SVM(
                    C=self.C,
                    kernel=self.kernel_name,
                    gamma=self.gamma_value,
                    degree=self.degree,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    learning_rate=self.learning_rate,
                    verbose=False
                )
                binary_svm._fit_binary(X, y_binary)
                self.models.append(binary_svm)
        
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train binary SVM
        
        Args:
            X: Training features
            y: Binary labels (-1 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        self.loss_history = []
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Compute loss
            loss = self._compute_hinge_loss(X, y)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                if self.verbose:
                    logger.info(f"Converged at iteration {iteration + 1}")
                break
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function for binary classification
        
        Args:
            X: Features
            
        Returns:
            Decision values
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if self.is_binary:
            decision = self._decision_function(X)
            # Convert back from -1/1 to 0/1
            return np.where(decision >= 0, 1, 0)
        else:
            # Multi-class: use one-vs-rest
            decisions = np.zeros((X.shape[0], self.n_classes))
            for i, model in enumerate(self.models):
                decisions[:, i] = model._decision_function(X)
            
            return np.argmax(decisions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (using Platt scaling approximation)
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities
        """
        if self.is_binary:
            decision = self._decision_function(X)
            # Use sigmoid for probability approximation
            proba_positive = utils.sigmoid(decision)
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Multi-class: use softmax on decision values
            decisions = np.zeros((X.shape[0], self.n_classes))
            for i, model in enumerate(self.models):
                decisions[:, i] = model._decision_function(X)
            
            return utils.softmax(decisions)
    
    def get_support_vectors(self, X: np.ndarray, y: np.ndarray, 
                           margin_threshold: float = 1.0) -> np.ndarray:
        """
        Get support vectors (samples near the decision boundary)
        
        Args:
            X: Training features
            y: Training labels
            margin_threshold: Threshold for determining support vectors
            
        Returns:
            Indices of support vectors
        """
        if not self.is_binary:
            logger.warning("Support vector extraction only supported for binary classification")
            return np.array([])
        
        y_binary = np.where(y == 0, -1, 1)
        decision = self._decision_function(X)
        margin = y_binary * decision
        
        # Support vectors are those with margin <= threshold
        support_vector_indices = np.where(margin <= margin_threshold)[0]
        
        return support_vector_indices
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (magnitude of weights)
        Works best with linear kernel
        
        Returns:
            Feature importance scores
        """
        if self.is_binary:
            return np.abs(self.weights)
        else:
            # For multi-class, return average importance across classes
            weights = np.array([model.weights for model in self.models])
            return np.mean(np.abs(weights), axis=0)
