import numpy as np

class KNN:
    """
    Implements the K-Nearest Neighbors (KNN) classification algorithm.
    
    Attributes:
        k: Number of nearest neighbors used to make a prediction.
        X_train: NumPy array containing training features.
        y_train: NumPy array containing training labels.
    """
    def __init__(self, k: int):
        """
        Initializes the KNN classifier.

        Args:
            k: Number of nearest neighbors used for classification.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def predict(self, X: np.ndarray) -> np.ndarray: 
        """
        Predicts the class labels for a set of data points.
        
        Args:
            X: NumPy array containing the data points to label.
        
        Returns:
            NumPy array of the predicted class labels for each point.
        """
        y_pred = np.empty(len(X), dtype=int)
        euclidean_dist = np.zeros(len(self.X_train))

        for point in range(len(X)):

            labels = []

            for index in range(len(euclidean_dist)):
                euclidean_dist[index] = np.linalg.norm(self.X_train[index] - X[point])

            distance_sorted = np.argsort(euclidean_dist)

            for count in range(self.k): 
                labels.append(self.y_train[distance_sorted[count]])

            predicted_label = np.bincount(labels).argmax()
            y_pred[point] = predicted_label

        return y_pred


    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits the KNN model using the given training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train