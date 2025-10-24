"""
Support Vector Machine (SVM) implementation from scratch
Supports multiple kernel functions
"""

import numpy as np
from typing import Callable, Optional
import logging
import utils
import config
# Optional kernel approximation imports
try:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.preprocessing import PolynomialFeatures
except Exception:
    RBFSampler = None
    PolynomialFeatures = None

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
                 learning_rate: float = 0.001, batch_size: int = 32,
                 optimization: str = 'gradient_descent', verbose: bool = True):
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
        self.batch_size = batch_size
        self.optimization = optimization
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.is_binary = None
        self.models = []  # For multi-class (one-vs-rest)
        self.loss_history = []
        self.gamma_value = None
        # Feature map for kernel approximation (used for SGD/mini-batch with non-linear kernels)
        self.feature_map = None
        
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
            # If using non-linear kernel with SGD/mini-batch, use kernel approximation mapping
            if self.kernel_name in ('rbf', 'polynomial') and self.optimization in ('sgd', 'mini_batch'):
                if self.kernel_name == 'rbf':
                    if RBFSampler is None:
                        logger.warning('RBFSampler not available; falling back to full-batch kernel computations')
                        self._fit_binary(X, y_binary)
                        return self
                    n_components = config.SVM.get('rff_components', 200)
                    self.feature_map = RBFSampler(gamma=self.gamma_value, n_components=n_components, random_state=config.RANDOM_STATE)
                    X_mapped = self.feature_map.fit_transform(X)
                else:  # polynomial
                    if PolynomialFeatures is None:
                        logger.warning('PolynomialFeatures not available; falling back to full-batch kernel computations')
                        self._fit_binary(X, y_binary)
                        return self
                    # Be cautious: polynomial expansion can be large; respect degree
                    poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                    self.feature_map = poly
                    X_mapped = self.feature_map.fit_transform(X)

                # Train in the mapped feature space using linear SVM updates
                self._fit_binary(X_mapped, y_binary)
            else:
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
                    batch_size=self.batch_size,
                    optimization=self.optimization,
                    verbose=False
                )
                # Use fit so each binary SVM can apply kernel approximation if requested
                binary_svm.fit(X, y_binary)
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
        
        # Training loop with support for SGD and mini-batch (linear kernel only)
        for iteration in range(self.max_iterations):
            if self.optimization == 'gradient_descent':
                # Full-batch
                loss = self._compute_hinge_loss(X, y)
                self.loss_history.append(loss)

                dw, db = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            elif self.optimization == 'sgd':
                # Stochastic gradient descent (per-sample updates)
                # If a feature_map was used, the input X is already mapped and SGD is supported
                if (self.kernel_name != 'linear') and (self.feature_map is None):
                    logger.warning('SGD is supported efficiently only for linear kernel (or when kernel approximation is used); falling back to full-batch')
                    loss = self._compute_hinge_loss(X, y)
                    self.loss_history.append(loss)
                    dw, db = self._compute_gradients(X, y)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                else:
                    indices = np.random.permutation(n_samples)
                    total_loss = 0.0
                    for i in indices:
                        Xi = X[i:i+1]
                        yi = y[i:i+1]
                        loss_i = self._compute_hinge_loss(Xi, yi)
                        total_loss += loss_i
                        dw, db = self._compute_gradients(Xi, yi)
                        self.weights -= self.learning_rate * dw
                        self.bias -= self.learning_rate * db
                    loss = total_loss / n_samples
                    self.loss_history.append(loss)

            elif self.optimization == 'mini_batch':
                # Mini-batch gradient descent
                # Allow mini-batch when kernel approximation (feature_map) was applied
                if (self.kernel_name != 'linear') and (self.feature_map is None):
                    logger.warning('Mini-batch is supported efficiently only for linear kernel (or when kernel approximation is used); falling back to full-batch')
                    loss = self._compute_hinge_loss(X, y)
                    self.loss_history.append(loss)
                    dw, db = self._compute_gradients(X, y)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                else:
                    indices = np.random.permutation(n_samples)
                    total_loss = 0.0
                    n_batches = 0
                    for start_idx in range(0, n_samples, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, n_samples)
                        batch_idx = indices[start_idx:end_idx]
                        X_batch = X[batch_idx]
                        y_batch = y[batch_idx]
                        loss_b = self._compute_hinge_loss(X_batch, y_batch)
                        total_loss += loss_b
                        n_batches += 1
                        dw, db = self._compute_gradients(X_batch, y_batch)
                        self.weights -= self.learning_rate * dw
                        self.bias -= self.learning_rate * db
                    loss = total_loss / max(1, n_batches)
                    self.loss_history.append(loss)

            else:
                raise ValueError(f"Unknown optimization method: {self.optimization}")

            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {self.loss_history[-1]:.6f}")

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
        # If a feature map (kernel approximation) was used during training, apply it here
        if getattr(self, 'feature_map', None) is not None:
            try:
                X_trans = self.feature_map.transform(X)
            except Exception:
                # Fallback to fit_transform if transform not available
                X_trans = self.feature_map.fit_transform(X)
            return np.dot(X_trans, self.weights) + self.bias

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
