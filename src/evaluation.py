import numpy as np
from typing import Callable

def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy of predicted labels.
    
    Args:
        y: NumPy array containing the actual labels.
        y_pred: NumPy array containing the predicted labels.

    Returns:
        The proportion of correctly predicted labels.
    """
    return np.sum(y == y_pred) / len(y)


def cross_validation(X_train: np.ndarray, y_train: np.ndarray, num_folds: int, 
                     algorithm: type, model_params: dict, scaler_class: type) -> list[float]:
    """
    Evaluates a classification model using K-fold cross-validation.

    Args:
        X_train: NumPy array containing the training features.
        y_train: NumPy array containing the corresponding training labels.
        num_folds: Number of folds used for cross-validation.
        algorithm: Classification model class to evaluate.
        model_params: Dictionary containing the model's hyperparameters.
        scaler_class: Feature scaler class used to scale the data.

    Returns:
        A list containing the accuracy of each fold, followed by the average accuracy.
    """
    list_of_accuracies = []
    total_accuracy = 0

    X_folds = np.array_split(X_train, num_folds)
    y_folds = np.array_split(y_train, num_folds)

    for index in range(num_folds):

        X_validation_set = X_folds[index]
        y_validation_set = y_folds[index]

        X_filtered_arrays = [arr for i, arr in enumerate(X_folds) if i != index]
        X_training_set = np.concatenate(X_filtered_arrays)

        scaler = scaler_class()
        X_training_set = scaler.fit_transform(X_training_set)
        X_validation_set = scaler.transform(X_validation_set)

        y_filtered_arrays = [arr for i, arr in enumerate(y_folds) if i != index]
        y_training_set = np.concatenate(y_filtered_arrays)

        model = algorithm(**model_params)
        model.fit(X_training_set, y_training_set)
        y_pred = model.predict(X_validation_set)

        fold_accuracy = accuracy(y_validation_set, y_pred)
        list_of_accuracies.append(fold_accuracy)
        total_accuracy += fold_accuracy
    
    # Calculate the average accuracy across all folds.
    list_of_accuracies.append(total_accuracy / num_folds)
    
    return list_of_accuracies