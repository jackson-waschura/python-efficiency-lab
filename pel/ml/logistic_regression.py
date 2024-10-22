"""
Logistic Regression is used for binary classification problems.

It uses the sigmoid function to map linear combinations of features to probabilities:
    sigmoid(z) = 1 / (1 + e^(-z))
    h(x) = sigmoid(theta^T * x)
    
The cost function for logistic regression is the log loss:
    J(theta) = -1/m * sum(y * log(h(x)) + (1-y) * log(1-h(x)))
where m is the number of samples and y is the binary target value.

Gradient descent is typically used to minimize the cost function:
    theta := theta - alpha * 1/m * sum((h(x) - y) * x)

From a probabilistic standpoint, we are trying to maximize the likelihood of the data.
If we make the modelling assumption that y|x ~ Bernoulli(h(x)), then the likelihood of the data is:
    L(theta) = prod(h(x)^y * (1-h(x))^(1-y))
Taking the log of the likelihood, we get the log loss function mentioned earlier.

Another way to think about Logistic Regression is that it is a linear model for the log-odds of the positive class:
    log(odds) = theta^T * x
    odds = exp(theta^T * x)
    prob(y=1|x) = odds / (1 + odds)
    prob(y=1|x) = 1 / (1 + 1/odds)
    prob(y=1|x) = 1 / (1 + exp(-theta^T * x))
This leads naturally to the sigmoid function instead of something like the hyperbolic tangent.

Also, this allows us to interpret the model in terms of the growth in odds (e.g. "each unit increase in x is associated
with an increase in the odds by a factor of exp(theta)").
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionNumpy:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate=0.01, n_iterations=1000):
        """
        Fit the model to the data using gradient descent.
        """
        assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same."
        assert X.shape[1] == self.n_features, "The number of features in X must match the model."

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        for _ in range(n_iterations):
            z = X @ self.theta
            h = self.sigmoid(z)
            gradient = X.T @ (h - y) / y.size
            self.theta -= learning_rate * gradient

    def predict_proba(self, X: np.ndarray):
        """
        Predict the probability of the positive class for the given features.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(X @ self.theta)

    def predict(self, X: np.ndarray, threshold=0.5):
        """
        Predict the class for the given features.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, X: torch.Tensor, y: torch.Tensor, learning_rate=0.01, n_iterations=1000):
        """
        Fit the model to the data using gradient descent.
        """
        assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same."
        assert X.shape[1] == self.linear.in_features, "The number of features in X must match the model."

        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        for _ in range(n_iterations):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = criterion(y_pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    def predict_proba(self, X: torch.Tensor):
        """
        Predict the probability of the positive class for the given features.
        """
        with torch.no_grad():
            return self(X)

    def predict(self, X: torch.Tensor, threshold=0.5):
        """
        Predict the class for the given features.
        """
        return (self.predict_proba(X) >= threshold).int()


if __name__ == "__main__":
    # Generate a dataset
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_test = np.array([[1.0, 1.0], [-1.0, -1.0]])

    # Numpy model
    model_np = LogisticRegressionNumpy(2)
    model_np.fit(X, y)

    # PyTorch model
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    model_torch = LogisticRegressionTorch(2)
    model_torch.fit(X_torch, y_torch)

    print("Numpy model predictions:")
    print("Probabilities:", model_np.predict_proba(X_test))
    print("Classes:", model_np.predict(X_test))

    print("\nPyTorch model predictions:")
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    print("Probabilities:", model_torch.predict_proba(X_test_torch).detach().numpy().squeeze())
    print("Classes:", model_torch.predict(X_test_torch).detach().numpy().squeeze())

