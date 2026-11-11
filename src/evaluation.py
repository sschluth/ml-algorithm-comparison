import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def cross_validation(X_train: np.ndarray, y_train: np.ndarray, num_folds: int,
                     algorithm: type, model_params: dict, scaler_class: type) -> dict:
    """
    Evaluates a classification model using stratified K-fold cross-validation.

    Assumptions:
        - The number of folds is between 2 and the number of training samples.
        - The feature and label arrays contain the same number of samples.
        - Each class has at least num_folds samples.

    Args:
        X_train: NumPy array containing the training features.
        y_train: NumPy array containing the corresponding training labels.
        num_folds: Number of folds used for cross-validation.
        algorithm: Classification model class to evaluate.
        model_params: Dictionary containing the model's hyperparameters.
        scaler_class: Feature scaler class used to scale the data.

    Returns:
        A dictionary containing each metric's score for every fold and its average,
        along with the combined confusion matrix across all folds.
    """
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    total_confusion_matrix = None

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)

    # Generates stratified training and validation indices for each fold.
    for train_index, validation_index in skf.split(X_train, y_train):
        X_training_set = X_train[train_index]
        y_training_set = y_train[train_index]

        X_validation_set = X_train[validation_index]
        y_validation_set = y_train[validation_index]

        scaler = scaler_class()

        # Fits the scaler using only the training fold to prevent data leakage.
        X_training_set = scaler.fit_transform(X_training_set)

        # Applies the training fold's scaling parameters to the validation fold.
        X_validation_set = scaler.transform(X_validation_set)

        model = algorithm(**model_params)
        model.fit(X_training_set, y_training_set)
        y_pred = model.predict(X_validation_set)

        # Calculates the evaluation metrics for the current fold.
        metrics["accuracy"].append(
            accuracy_score(y_validation_set, y_pred)
        )
        metrics["precision"].append(
            precision_score(y_validation_set, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"].append(
            recall_score(y_validation_set, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1"].append(
            f1_score(y_validation_set, y_pred, average="weighted", zero_division=0)
        )

        # Adds the current fold's confusion matrix to the combined matrix.
        fold_confusion_matrix = confusion_matrix(y_validation_set, y_pred)

        if total_confusion_matrix is None:
            total_confusion_matrix = fold_confusion_matrix
        else:
            total_confusion_matrix += fold_confusion_matrix

    # Calculates the average score for each metric across all folds.
    for metric in metrics:
        metrics[metric].append(np.mean(metrics[metric]))

    metrics["confusion_matrix"] = total_confusion_matrix

    return metrics