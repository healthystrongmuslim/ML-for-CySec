import numpy as np
from typing import Optional

class RBFSampler:
    """
    Minimal Random Fourier Features (RFF) implementation to approximate the RBF kernel.
    This implementation is compatible with the basic API of sklearn's RBFSampler.
    Kernel: k(x, y) = exp(-gamma * ||x - y||^2)

    Parameters
    ----------
    gamma : float
        Kernel coefficient for the RBF kernel.
    n_components : int
        Number of Monte-Carlo samples to approximate the kernel. This determines
        the dimensionality of the transformed feature space.
    random_state : Optional[int]
        Seed for the random number generator for reproducibility.
    """
    def __init__(self, gamma: float = 1.0, n_components: int = 100, random_state: Optional[int] = None):
        self.gamma = gamma
        self.n_components = int(n_components)
        self.random_state = random_state
        self.random_weights_ = None
        self.random_offset_ = None

    def fit(self, X, y=None):
        """
        Fit the model with random weights and offsets.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.gamma is None:
            raise ValueError("gamma must be set for RBFSampler")

        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        # Draw random weights from a Gaussian distribution
        # The scaling factor is sqrt(2 * gamma)
        scale = np.sqrt(2.0 * float(self.gamma))
        self.random_weights_ = rng.normal(loc=0.0, scale=scale, size=(n_features, self.n_components))

        # Draw random offsets from a uniform distribution on [0, 2*pi]
        self.random_offset_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        
        return self

    def transform(self, X):
        """
        Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        if self.random_weights_ is None or self.random_offset_ is None:
            raise ValueError("RBFSampler has not been fitted. Call fit() first.")
        
        # Project the data onto the random weights
        projection = X.dot(self.random_weights_) + self.random_offset_
        
        # Apply cosine transformation and scale
        np.cos(projection, out=projection)
        projection *= np.sqrt(2.0 / self.n_components)
        
        return projection

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y).transform(X)
