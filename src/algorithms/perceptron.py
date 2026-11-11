import numpy as np

class Perceptron:
    """
    Implements a multiclass Perceptron

    Assumptions:
        - Training data contains at least two unique classes.
        - Input features contain no missing or invalid values.
        - All training samples have the same number of features.

    Attributes:
        learning_rate: Controls the magnitude of the weight updates.
        epochs: Maximum number of training iterations.
        X_train: NumPy array containing the training features.
        y_train: NumPy array containing the training labels.
        weights: Model weights learned during training.
        classes: NumPy array containing the unique class labels.
    """
    def __init__(self, learning_rate: float, epochs: int):
        """
        Initializes the multiclass Perceptron classifier
        
        Args:
            learning_rate: Controls the magnitude of the weight updates.
            epochs: Number of training iterations.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.X_train = None
        self.y_train = None
        self.weights = None
        self.classes = None


    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """
        Predicts the class label(s) of the given input data.

        Args:
            X: Numpy array containing the input features.

        Returns:
            The predicted class label(s)
        """
        if (X.ndim == 1):
            y_pred_index = np.argmax(self._forward(X))
            return self.classes[y_pred_index]

        else:
            y_pred_indices = np.argmax(np.dot(self.weights, X.T), axis=0)
            return self.classes[y_pred_indices]

        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits the Perceptron model using the given training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

        # Initializes a weight vector for each class.
        self.weights = np.zeros((len(self.classes), len(X_train[0])), dtype=float)

        misclassified = True
        epoch = 0

        # Stops training early if an entire epoch passes without misclassifications.
        while (epoch < self.epochs and misclassified):
            misclassified = False
            for i in range(len(self.X_train)):
                x = self.X_train[i]
                y_true = self.y_train[i]

                y_pred_index = np.argmax(self._forward(x))
                y_pred = self.classes[y_pred_index]

                if (y_pred != y_true):
                    misclassified = True
                    self._update(x, y_true, y_pred)

            epoch += 1


    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Private helper method to calculate the predicted label scores.

        Args:
            x: NumPy array containing the features of the given data point.

        Returns:
            NumPy array containing the scores of the labels
        """
        return np.dot(self.weights, x)


    def _update(self, x: np.ndarray, y_true: int, y_pred: int) -> None:
        """
        Private helper method to update the weights.

        Args:
            x: NumPy array containing the features of the given data point.
            y_true: The actual class label.
            y_pred: The predicted class label.
        """
        y_true_index = np.where(self.classes == y_true)[0][0]
        y_pred_index = np.where(self.classes == y_pred)[0][0]

        # Increases the weights for the correct class and decreases the weights
        # for the incorrectly predicted class.
        self.weights[y_true_index] += self.learning_rate * x
        self.weights[y_pred_index] -= self.learning_rate * x