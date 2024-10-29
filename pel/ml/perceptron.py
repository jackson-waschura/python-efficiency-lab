"""
The Perceptron is a simple binary classifier that predicts the label of a data point by looking at the weighted sum
of the input features and applying a step function to the result.

It is essentially a single layer neural network with a step activation function.

The prediction is made using:
    y_pred = step(w @ x + b)
    where step(z) = 1 if z ≥ 0, -1 otherwise

The update rule for the Perceptron can be written uniformly for all samples as:
    delta = y_true - y_pred
    w = w + (delta * x) / 2
    b = b + delta / 2

where:
- y_true is the true label of the data point (typically +1 or -1)
- y_pred is the predicted label
- w is the weight vector
- x is the input feature vector
- b is the bias term
- delta will be:
    * 0 when correctly classified (y_true = y_pred)
    * +2 when y_true = +1 and y_pred = -1 (false negative)
    * -2 when y_true = -1 and y_pred = +1 (false positive)

The algorithm converges when the data is linearly separable, guaranteeing to find a separating hyperplane
in a finite number of updates.
"""

from sklearn.utils import shuffle
import numpy as np


class Perceptron:
    def __init__(self, n_features: int):
        """Initialize a Perceptron classifier.

        Args:
            n_features: Number of input features
        """
        self.n_features = n_features

        # Initialize weights and bias
        self.initialize_weights()
    
    def initialize_weights(self):
        self.w = np.random.randn(self.n_features)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Args:
            X: Array-like of shape (n_samples, n_features)
                The input samples to predict

        Returns:
            Array of shape (n_samples,) containing predicted labels (-1 or 1)
        """
        # Ensure X is a 2D array
        X = np.atleast_2d(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but Perceptron was initialized with {self.n_features} features"
            )
        
        return np.where(X @ self.w + self.b >= 0, 1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100, batch_size: int = 32, lr: float = 0.1) -> tuple[list, bool]:
        """Fit the perceptron model using minibatch updates.

        Args:
            X: Array-like of shape (n_samples, n_features)
                Training data
            y: Array-like of shape (n_samples,)
                Target labels (-1 or 1)
            max_iter: Maximum number of passes over the training data
            batch_size: Size of minibatches for updates
            lr: Learning rate for the updates
        Returns:
            errors: List of number of misclassifications in each epoch
            converged: Boolean indicating if the model converged
        """
        X = np.atleast_2d(X)
        y = np.asarray(y)
        n_samples = len(X)

        # Initialize weights and bias
        self.initialize_weights()
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but Perceptron was initialized with {self.n_features} features"
            )

        errors = []
        for epoch in range(max_iter):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            misclassified = 0
            
            # Process minibatches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Make predictions for the batch
                y_pred = self.predict(X_batch)
                
                # Calculate deltas for the batch
                deltas = y_batch - y_pred
                
                # Update weights using the mean of the batch updates
                # Shape of X_batch: (batch_size, n_features)
                # Shape of deltas: (batch_size,)
                # Broadcasting will multiply each feature by its corresponding delta
                self.w += lr * np.mean(deltas.reshape(-1, 1) * X_batch, axis=0) / 2
                self.b += lr * np.mean(deltas) / 2
                
                # Count misclassifications in this batch
                misclassified += np.sum(np.abs(deltas) > 0)
            
            errors.append(misclassified)
            
            # If no errors, we've converged
            if misclassified == 0:
                return errors, True
        
        return errors, False

if __name__ == "__main__":
    # Example usage
    perceptron = Perceptron(n_features=4)
    X = np.concatenate([np.random.randn(100, 4) + 1.5, np.random.randn(100, 4) - 1.5])
    y = np.concatenate([np.ones(100), -np.ones(100)])
    X, y = shuffle(X, y, random_state=42)
    errors, converged = perceptron.fit(X, y)
    print(errors, converged)