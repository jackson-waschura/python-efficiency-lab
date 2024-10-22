"""
Linear Regression involves finding the best-fit line for a set of points.

When the dataset is small enough to fit in memory, we can use the normal equation to find the best-fit line.
This looks like:
    theta = (X^T * X)^-1 * X^T * y
Where X is the matrix of features and y is the vector of target values.

When the dataset is too large to fit in memory, we can use stochastic gradient descent to find the best-fit line.
This looks like:
    theta := theta - alpha * (h(x) - y) * x
Where h(x) is the hypothesis function, alpha is the learning rate, and y is the target value.

Multicollinearity in your features is when two or more features are highly correlated.
This can cause problems with the normal equation because the matrix X^T * X is not invertible.
"""

import numpy as np
import torch

class LinearRegressionNumpy:
    def __init__(self, n_features: int, n_outputs: int):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.theta = np.zeros((n_features + 1, n_outputs))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data using the normal equation.
        """
        # Check that the shapes of X and y are correct
        assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same."
        assert X.shape[1] == self.n_features, "The number of features in X must be the same as the number of features in the model."
        assert y.shape[1] == self.n_outputs, "The number of outputs in y must be the same as the number of outputs in the model."

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray):
        """
        Predict the target values for the given features.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.theta

class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, n_features: int, n_outputs: int):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.linear = torch.nn.Linear(n_features, n_outputs)

    def forward(self, x):
        return self.linear(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the model to the data using batch gradient descent.

        Arguments:
            X (torch.Tensor with shape (n_samples, n_features)): The features of the data.
            y (torch.Tensor with shape (n_samples, n_outputs)): The target values of the data.
        """
        # Check that the shapes of X and y are correct
        assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same."
        assert X.shape[1] == self.n_features, "The number of features in X must be the same as the number of features in the model."
        assert y.shape[1] == self.n_outputs, "The number of outputs in y must be the same as the number of outputs in the model."

        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        for _ in range(1000):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor):
        """
        Predict the target values for the given features.
        """
        return self(X)


if __name__ == "__main__":
    # Generate a dataset
    true_theta = np.array([[1.0], [2.0]], dtype=np.float32)
    true_bias = np.array([[0.5]], dtype=np.float32)
    X = np.array([[1.0, 2.0], [2.5, 3.0], [3.0, 4.5], [4.0, 5.0]], dtype=np.float32)
    y = X @ true_theta + true_bias
    test_X = np.array([[1.0, 1.0], [7.0, 6.0]], dtype=np.float32)

    # Fit the model
    model_np = LinearRegressionNumpy(2, 1)
    model_np.fit(X, y)
    
    model_torch = LinearRegressionTorch(2, 1)
    model_torch.fit(torch.tensor(X), torch.tensor(y))

    print("True theta:", np.concatenate([true_bias, true_theta], axis=0).T)
    print("Numpy model theta:", model_np.theta.T)
    print("Torch model theta:", np.concatenate([model_torch.linear.bias.data.numpy().reshape(-1, 1), model_torch.linear.weight.data.numpy()], axis=1))
    print("")
    print("True values:", (test_X @ true_theta + true_bias).T)
    print("Numpy model predictions:", model_np.predict(test_X).T)
    print("Torch model predictions:", model_torch.predict(torch.tensor(test_X)).detach().cpu().numpy().T)
