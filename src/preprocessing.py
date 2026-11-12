import pandas as pd
import numpy as np


def load_data(dataset: str, y_col: str | int, x_cols: list[str | int],
              header: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a dataset, preprocesses categorical features when applicable,
    encodes labels, and separates the features (X) from the labels (y).

    Assumptions:
        - The dataset is a properly formatted CSV file.
        - The specified feature and label columns exist.
        - The dataset contains no missing or invalid values.
        - Student Performance grade values range from 0 to 20.
        - Student Performance categorical values match the defined mappings.
        - Non-Student Performance datasets must have numerical feature columns.

    Args:
        dataset: Path to the CSV file.
        y_col: Column name or index for the labels.
        x_cols: List of column names or indices for the features.
        header: Row index containing column names, or None if the
                dataset has no header.

    Returns:
        X: NumPy array containing the numerical features.
        y: NumPy array containing the encoded labels.
    """
    df = pd.read_csv(dataset, sep=None, header=header)

    y = df[y_col]

    if y_col == "G3":
        # Group final grades into four categories
        y = pd.cut(y, bins=[-1, 9, 13, 16, 20], labels=[0, 1, 2, 3])

        # Encode binary categorical features
        binary_mappings = {
            "school": {"GP": 0, "MS": 1},
            "sex": {"F": 0, "M": 1},
            "address": {"U": 0, "R": 1},
            "famsize": {"LE3": 0, "GT3": 1},
            "Pstatus": {"T": 0, "A": 1},
            "schoolsup": {"no": 0, "yes": 1},
            "famsup": {"no": 0, "yes": 1},
            "paid": {"no": 0, "yes": 1},
            "activities": {"no": 0, "yes": 1},
            "nursery": {"no": 0, "yes": 1},
            "higher": {"no": 0, "yes": 1},
            "internet": {"no": 0, "yes": 1},
            "romantic": {"no": 0, "yes": 1}
        }

        for col, mapping in binary_mappings.items():
            if col in x_cols:
                df[col] = df[col].map(mapping)

        # One-hot encode nominal categorical features
        categorical_cols = ["Mjob", "Fjob", "reason", "guardian"]
        categorical_cols = [col for col in categorical_cols if col in x_cols]

        X = pd.get_dummies(df[x_cols], columns=categorical_cols, dtype=int).to_numpy()

    else:
        # Encode categorical labels
        unique_y_values = y.unique()
        label_mapping = {label: i for i, label in enumerate(unique_y_values)}
        y = y.map(label_mapping)

        X = df[x_cols].to_numpy()

    y = y.to_numpy(dtype=int)

    return X, y

    
def split_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into 80% training data and 20% testing data using a reproducible random split.

    Assumptions:
        - X and y contain the same number of samples.
        - The dataset contains at least two samples.

    Args:
        X: NumPy array of features to be split.
        y: NumPy array of corresponding labels to be split.

Returns:
        X_train: Features used to train the models.
        y_train: Corresponding training labels.
        X_test: Features reserved for testing.
        y_test: Corresponding testing labels.
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

    Assumptions:
        - Input features are numerical.
        - No features have a constant value across the training data.
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

    Assumptions:
        - Input features are numerical.
        - No features have a standard deviation of zero.
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