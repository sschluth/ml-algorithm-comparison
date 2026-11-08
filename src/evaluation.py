import numpy as np

def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y == y_pred) / len(y)

def cross_validation(X_train: np.ndarray, y_train: np.ndarray, num_folds: int, algorithm: function, model_params: dict) -> list[float]:
    list_of_accuracies = []
    total_accuracy = 0
    X_folds = np.array_split(X_train, num_folds)
    y_folds = np.array_split(y_train, num_folds)

    for index in range(num_folds):
        X_validation_set = X_folds[index]
        y_validation_set = y_folds[index]

        X_filtered_arrays = [arr for i, arr in enumerate(X_folds) if i != index]
        X_training_set = np.concatenate(X_filtered_arrays)
        y_filtered_arrays = [arr for i, arr in enumerate(y_folds) if i != index]
        y_training_set = np.concatenate(y_filtered_arrays)

        model = algorithm(**model_params)
        model.fit(X_training_set, y_training_set)
        y_pred = model.predict(X_validation_set)

        fold_accuracy = accuracy(y_validation_set, y_pred)
        list_of_accuracies.append(fold_accuracy)
        total_accuracy += fold_accuracy

    list_of_accuracies.append(total_accuracy / num_folds)
    return list_of_accuracies