"""
Logistic Regression implementation from scratch
Supports binary and multi-class classification
"""

import numpy as np
from typing import Tuple, Optional
import logging
import utils
import config

logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch
    
    Supports:
    - Binary and multi-class classification
    - Gradient descent, SGD, and mini-batch training
    - L2 regularization
    - Gradient checking
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, lambda_: float = 0.01,
                 batch_size: int = 32, optimization: str = 'gradient_descent',
                 gradient_check: bool = False, verbose: bool = True):
        """
        Initialize Logistic Regression model
        
        Args:
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            lambda_: Regularization parameter
            batch_size: Batch size for mini-batch SGD
            optimization: 'gradient_descent', 'sgd', or 'mini_batch'
            gradient_check: Whether to perform gradient checking
            verbose: Whether to print training progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.optimization = optimization
        self.gradient_check = gradient_check
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.is_binary = None
        self.loss_history = []
        
    def _initialize_weights(self, n_features: int, n_classes: int) -> None:
        """Initialize weights and bias"""
        if self.is_binary:
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0.0
        else:
            self.weights = np.random.randn(n_features, n_classes) * 0.01
            self.bias = np.zeros(n_classes)
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss with L2 regularization
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels
            
        Returns:
            Loss value
        """
        n_samples = X.shape[0]
        
        if self.is_binary:
            z = np.dot(X, self.weights) + self.bias
            predictions = utils.sigmoid(z)
            
            # Binary cross-entropy
            epsilon = 1e-15  # For numerical stability
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Add L2 regularization
            loss += (self.lambda_ / (2 * n_samples)) * np.sum(self.weights ** 2)
        else:
            z = np.dot(X, self.weights) + self.bias
            predictions = utils.softmax(z)
            
            # Multi-class cross-entropy
            y_one_hot = utils.one_hot_encode(y, self.n_classes)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))
            
            # Add L2 regularization
            loss += (self.lambda_ / (2 * n_samples)) * np.sum(self.weights ** 2)
        
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Labels
            
        Returns:
            Gradients for weights and bias
        """
        n_samples = X.shape[0]
        
        if self.is_binary:
            z = np.dot(X, self.weights) + self.bias
            predictions = utils.sigmoid(z)
            
            # Gradient computation
            dz = predictions - y
            dw = (1 / n_samples) * np.dot(X.T, dz) + (self.lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(dz)
        else:
            z = np.dot(X, self.weights) + self.bias
            predictions = utils.softmax(z)
            
            # Gradient computation
            y_one_hot = utils.one_hot_encode(y, self.n_classes)
            dz = predictions - y_one_hot
            dw = (1 / n_samples) * np.dot(X.T, dz) + (self.lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(dz, axis=0)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        self.is_binary = (self.n_classes == 2)
        
        # Initialize weights
        self._initialize_weights(n_features, self.n_classes)
        
        # Gradient checking (only once at start)
        if self.gradient_check:
            self._perform_gradient_check(X[:min(10, n_samples)], y[:min(10, n_samples)])
        
        # Training loop
        for iteration in range(self.max_iterations):
            if self.optimization == 'gradient_descent':
                # Full batch gradient descent
                loss = self._compute_loss(X, y)
                dw, db = self._compute_gradients(X, y)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
            elif self.optimization == 'sgd':
                # Stochastic gradient descent
                indices = np.random.permutation(n_samples)
                total_loss = 0
                
                for i in indices:
                    Xi = X[i:i+1]
                    yi = y[i:i+1]
                    
                    loss = self._compute_loss(Xi, yi)
                    total_loss += loss
                    
                    dw, db = self._compute_gradients(Xi, yi)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                
                loss = total_loss / n_samples
                
            elif self.optimization == 'mini_batch':
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                total_loss = 0
                n_batches = 0
                
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    loss = self._compute_loss(X_batch, y_batch)
                    total_loss += loss
                    n_batches += 1
                    
                    dw, db = self._compute_gradients(X_batch, y_batch)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                
                loss = total_loss / n_batches
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization}")
            
            self.loss_history.append(loss)
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                if self.verbose:
                    logger.info(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities
        """
        if self.is_binary:
            z = np.dot(X, self.weights) + self.bias
            proba = utils.sigmoid(z)
            return np.column_stack([1 - proba, proba])
        else:
            z = np.dot(X, self.weights) + self.bias
            return utils.softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def _perform_gradient_check(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform gradient checking to validate implementation
        
        Args:
            X: Small subset of features for checking
            y: Corresponding labels
        """
        logger.info("Performing gradient checking...")
        
        # Save current weights
        original_weights = self.weights.copy()
        original_bias = self.bias
        
        # Compute analytic gradients
        dw_analytic, db_analytic = self._compute_gradients(X, y)
        
        # Check weight gradients
        def loss_func(w):
            self.weights = w.reshape(original_weights.shape)
            return self._compute_loss(X, y)
        
        error = utils.gradient_check(
            loss_func,
            original_weights.flatten(),
            dw_analytic.flatten(),
            epsilon=config.LOGISTIC_REGRESSION['gradient_check_epsilon']
        )
        
        logger.info(f"Gradient check relative error: {error:.10f}")
        
        if error < 1e-5:
            logger.info("✓ Gradient check passed!")
        else:
            logger.warning("✗ Gradient check failed. Check your implementation.")
        
        # Restore original weights
        self.weights = original_weights
        self.bias = original_bias
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (magnitude of weights)
        
        Returns:
            Feature importance scores
        """
        if self.is_binary:
            return np.abs(self.weights)
        else:
            # For multi-class, return average importance across classes
            return np.mean(np.abs(self.weights), axis=1)
