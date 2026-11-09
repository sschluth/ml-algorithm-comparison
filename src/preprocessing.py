import pandas as pd
import numpy as np

def load_data(dataset: str, y_col:int, x_cols:list, header:int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a dataset and separates its features (X) and labels (y).

    Args:
        dataset: Path to CSV file.
        y_col: Column index for the labels.
        x_cols: List of column indices for the features.
        header: Row index containing column names, or None if the dataset has no header
    
    Returns:
        X: NumPy array containing the data points and their features
        y: NumPy array containing the corresponding labels
    """

    df = pd.read_csv(dataset, header=header)

    y = df.iloc[:, y_col]
    unique_y_values = y.unique()
    for i in range(len(unique_y_values)):
        y = y.replace(unique_y_values[i], i)

    X = df.iloc[:, x_cols].to_numpy()
    y = y.to_numpy(dtype=int)

    return X, y

    
def split_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits into 80% training data and 20% testing data using a random split

    Args:
        X: NumPy array of features to be split.
        y: NumPy array of corresponding labels to be split.

    Returns:
        X_train: NumPy array containing the features used to train the model.
        y_train: NumPy array containing the expected labels corresponding to the training features.
        X_test: NumPy array containing the unseen testing features.
        y_test: NumPy array containing the corresponding labels of the unseen testing features.
    """
    indices = np.arange(len(X))
    np.random.seed(15)
    np.random.shuffle(indices)

    split_index = int(len(X) * 0.8)
    train_index = indices[:split_index]
    test_index = indices[split_index:]

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test



class CustomMinMaxScaler:
    """
    Implements Min-Max feature scaling.
    """
    def __init__(self):
        """
        Initializes the minimum and maximum scaling parameters
        """
        self.x_min = None
        self.x_max = None

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Scales the training features using Min-Max Scaling.
        
        Args:
            X_train: NumPy array containing the training features.
        
        Returns:
            NumPy array containing the scaled training features.
        """
        self.x_min = X_train.min(axis=0)
        self.x_max = X_train.max(axis=0)

        if np.any(self.x_min == self.x_max):
            raise ValueError("Cannot scale constant features.")

        return (X_train - self.x_min) / (self.x_max - self.x_min)


    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transforms new data using the minimum and maximum values
        learned from the training data.
        
        Args:
            X_test: NumPy array containing the testing features to be transformed

        Returns:
            NumPy array of the transformed testing features.
        """
        return (X_test - self.x_min) / (self.x_max - self.x_min)



class CustomStandardScaler:
    """
    Implements standardization using z-score scaling.
    """
    def __init__(self):
        """
        Initializes the mean and standard deviation scaling parameters
        """
        self.mean = None
        self.std = None

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Scales the training features using Standardization (z-score scaling).
        
        Args:
            X_train: NumPy array containing the training features.
        
        Returns:
           NumPy array containing the scaled training features.
        """
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)

        if np.any(self.std == 0):
            raise ValueError("Cannot scale constant features.")
        
        return (X_train - self.mean) / self.std


    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transforms new data using the mean and standard deviation values
        learned from the training data.
        
        Args:
            X_test: NumPy array containing the testing features to be transformed

        Returns:
            NumPy array of the transformed testing features.
        """
        return (X_test - self.mean) / self.std