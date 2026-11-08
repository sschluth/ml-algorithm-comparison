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

    def predict(self, point: np.ndarray) -> int: 
        """
        Predicts the class label of a given data point.
        
        Args:
            point: NumPy array containing the features of the data point.
        
        Returns:
            Predicted class label.
        """
        labels = []
        euclidean_dist = np.zeros(len(self.X_train))

        for index in range(len(euclidean_dist)):
            euclidean_dist[index] = np.linalg.norm(self.X_train[index] - point)

        distance_sorted = np.argsort(euclidean_dist)

        for count in range(self.k): 
            labels.append(self.y_train[distance_sorted[count]])

        predicted_label = np.bincount(labels).argmax()
        return predicted_label

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits the KNN model using the given training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train