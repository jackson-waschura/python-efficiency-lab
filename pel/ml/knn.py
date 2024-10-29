"""
K Nearest Neighbors is a simple algorithm that predicts the label of a data point by looking at the labels of the k-nearest data points in the training set.

KNN can be used for both classification and regression tasks.
"""

import numpy as np
from scipy.stats import mode

class KNNRegressor:
    def __init__(self, k: int):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store the training data.

        Arguments:
            X (np.ndarray): The features of the training data.
            y (np.ndarray): The target values of the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given features.

        Arguments:
            X (np.ndarray): The features of the data to predict.

        Returns:
            np.ndarray: The predicted target values.
        """
        # Vectorized prediction calculation
        distances = np.linalg.norm(self.X_train[:, None, :] - X[None, :, :], axis=-1)
        k_indices = np.argsort(distances, axis=0)[:self.k]
        k_nearest_values = self.y_train[k_indices]
        predictions = np.mean(k_nearest_values, axis=0)
        return predictions

class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store the training data.

        Arguments:
            X (np.ndarray): The features of the training data.
            y (np.ndarray): The labels of the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given features.

        Arguments:
            X (np.ndarray): The features of the data to predict.

        Returns:
            np.ndarray: The predicted labels.
        """
        # Vectorized distance calculation
        distances = np.linalg.norm(self.X_train[:, None, :] - X[None, :, :], axis=-1)
        k_indices = np.argsort(distances, axis=0)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Vectorized majority vote
        most_common = mode(k_nearest_labels, axis=0).mode
        return most_common.flatten()

if __name__ == "__main__":
    # Example usage
    # Generate a simple dataset
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_train_reg = np.array([2.5, 3.5, 4.5, 5.5])
    y_train_clf = np.array([0, 1, 0, 1])

    # Test data
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

    # Expected results for comparison
    expected_reg = np.array([3.0, 4.5])
    expected_clf = np.array([0, 0])

    # KNN Regressor
    knn_regressor = KNNRegressor(k=2)
    knn_regressor.fit(X_train, y_train_reg)
    reg_predictions = knn_regressor.predict(X_test)
    print("KNN Regressor predictions:", reg_predictions)
    print("Expected Regressor results:", expected_reg)
    print("Regressor predictions match expected:", np.allclose(reg_predictions, expected_reg))

    # KNN Classifier
    knn_classifier = KNNClassifier(k=2)
    knn_classifier.fit(X_train, y_train_clf)
    clf_predictions = knn_classifier.predict(X_test)
    print("KNN Classifier predictions:", clf_predictions)
    print("Expected Classifier results:", expected_clf)
    print("Classifier predictions match expected:", np.array_equal(clf_predictions, expected_clf))
