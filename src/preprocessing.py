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
    y = y.to_numpy()

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


def min_max_scale(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales the training features using Min-Max Scaling.
    
    Args:
        X_train: NumPy array containing the training features.
    
    Returns:
        X_train_scaled: NumPy array containing the scaled training features.
        x_min: Minimum value of the training features.
        x_max: Maximum value of the training features.
    """
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)

    # Skip scaling if all features have zero range.
    if np.all(x_min == x_max):
        return X_train, x_min, x_max
    
    X_train_scaled = (X_train - x_min) / (x_max - x_min)
    return X_train_scaled, x_min, x_max


def z_scale(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales the training features using Standardization (z-score scaling).
    
    Args:
        X_train: NumPy array containing the training features.
    
    Returns:
        X_train_scaled: NumPy array containing the scaled training features.
        mean: The mean of the training features.
        std: The standard deviation of the training features.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Skip scaling if all features have zero standard deviation.
    if np.all(std == 0):        
        return X_train, mean, std
    
    X_train_scaled = (X_train - mean) / std

    return X_train_scaled, mean, std

def transform(X_test: np.ndarray, scale_param1: np.ndarray, scale_param2: np.ndarray, is_min_max: bool) -> np.ndarray:
    """
    Transforms the testing features using the parameters of the training features
    
    Args:
        X_test: NumPy array containing the testing features to be transformed
        scale_param1: Minimum value or the mean of the training features, depending on scaling method.
        scale_param2: Maximum value or the standard deviation of the training features, depending on scaling method.
        is_min_max: If True, applies Min-Max scaling; otherwise, applies Z-score scaling.

    Returns:
        X_test_scaled: NumPy array of the transformed testing features.
    """
    if is_min_max:

        # Skip transformation if the training features have zero range
        if np.all(scale_param1 == scale_param2):
            return X_test
        
        X_test_scaled = (X_test - scale_param1) / (scale_param2 - scale_param1)

    else:

        # Skip transformation if the training features have zero standard deviation
        if np.all(scale_param2 == 0):
            return X_test
        
        X_test_scaled = (X_test - scale_param1) / scale_param2

    return X_test_scaled