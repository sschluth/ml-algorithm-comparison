import numpy as np

class Logistic_Regression:
    """
    Implements a Logistic Regression classifier for binary and multiclass
    classification using batch gradient descent.
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, epoch: int):
        """
        Initializes the Logistic Regression classifier.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            learning_rate: Controls the magnitude of the weight and bias updates.
            epoch: Number of training iterations.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.num_of_classes = len(np.unique(self.y_train))

        # Initializes weights and bias based on the number of classes.
        if (self.num_of_classes > 2):
            self.weights = np.zeros((self.num_of_classes, len(X_train[0])))
            self.bias = np.zeros(self.num_of_classes)
        else:
            self.weights = np.zeros(len(X_train[0]))
            self.bias = 0


    def logistic_regression(self) -> None:
        """
        Trains the Logistic Regression model using batch gradient descent.
        Selects binary or multinomial logistic regression based on the number
        of classes.
        """
        if self.num_of_classes > 2:
            self._multinomial_logistic_regression()

        else:
            self._binary_logistic_regression()

     
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class label of the given input data.
        
        Args:
            X: NumPy array containing the input features.
        
        Returns:
            The predicted class labels.
        """
        if self.num_of_classes > 2:
            z = np.dot(X, self.weights.T) + self.bias
            y_pred = self._softmax(z)
            return np.argmax(y_pred, axis=-1)
        else:   
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            return np.array([1 if i > 0.5 else 0 for i in y_pred])

    
    def _multinomial_logistic_regression(self) -> None:
        """
        Trains the multinomial Logistic Regression model using
        batch gradient descent.
        """

        # Converts class labels into one-hot encoded vectors
        y_train_one_hot = np.eye(self.num_of_classes)[self.y_train]
        
        for _ in range(self.epoch):
            z = np.dot(self.X_train, self.weights.T) + self.bias
            y_pred = self._softmax(z)
            derivative_weights = (1/len(y_train_one_hot)) * np.dot((y_pred - y_train_one_hot).T, self.X_train)
            derivative_bias = (1/len(y_train_one_hot)) * np.sum(y_pred - y_train_one_hot, axis=0)
            self.weights -= self.learning_rate * derivative_weights
            self.bias -= self.learning_rate * derivative_bias

        
    def _binary_logistic_regression(self) -> None:
        """
        Trains the binary Logistic Regression model using
        batch gradient descent.
        """
        for _ in range(self.epoch):
            z = np.dot(self.X_train, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            derivative_weights = (1/len(self.y_train)) * np.dot((y_pred - self.y_train), self.X_train)
            derivative_bias = (1/len(self.y_train)) * np.sum(y_pred - self.y_train)
            self.weights -= self.learning_rate * derivative_weights
            self.bias -= self.learning_rate * derivative_bias

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function.

        Args:
            z: NumPy array containing the input values.

        Returns:
            NumPy array containing the sigmoid probabilities.
        """
        return 1/(1 + np.exp(-z))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function.

        Args:
            z: NumPy array containing the input values.

        Returns:
            NumPy array containing the normalized class probabilities.
        """
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)