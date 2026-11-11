import preprocessing
import evaluation
import algorithms.knn as knn
import algorithms.logistic_regression as logistic_regression
import algorithms.perceptron as perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def print_results(model_name: str, results: dict, num_folds: int):
    """
    Prints the cross-validation results for a classification model.

    Displays the accuracy, precision, recall, and F1-score for each fold
    and their averages, along with the combined confusion matrix.

    Args:
        model_name: Name of the model being evaluated.
        results: Dictionary containing evaluation metrics and confusion matrix.
        num_folds: Number of cross-validation folds.
    """
    print(f"\n{'=' * 50}")
    print(f"{model_name:^50}")
    print(f"{'=' * 50}")

    print(f"{'Metric':<15}", end="")
    for i in range(1, num_folds + 1):
        print(f"{'Fold ' + str(i):>8}", end="")
    print(f"{'Average':>10}")

    print("-" * 65)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = results[metric]
        print(f"{metric.capitalize():<15}", end="")

        for score in scores[:-1]:
            print(f"{score * 100:>7.2f}%", end="")

        print(f"{scores[-1] * 100:>9.2f}%")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


def main():
    """
    Runs the machine learning experiments.

    Loads and splits the dataset, configures the custom and scikit-learn
    models, evaluates each model using stratified K-fold cross-validation,
    and prints the resulting evaluation metrics.
    """

    # Experiment settings
    num_folds = 5
    knn_k = 5
    learning_rate = 0.05
    epochs = 5

    # Configure feature scalers for each algorithm
    custom_knn_scaler = preprocessing.CustomMinMaxScaler
    sklearn_knn_scaler = MinMaxScaler

    custom_log_reg_scaler = preprocessing.CustomStandardScaler
    sklearn_log_reg_scaler = StandardScaler

    custom_perceptron_scaler = preprocessing.CustomStandardScaler
    sklearn_perceptron_scaler = StandardScaler

    # Configure model hyperparameters
    custom_knn_params = {"k": knn_k}
    sklearn_knn_params = {"n_neighbors": knn_k}

    custom_log_reg_params = {"learning_rate": learning_rate, "epochs": epochs}
    sklearn_log_reg_params = {"max_iter": 1000}

    custom_perceptron_params = {"learning_rate": learning_rate, "epochs": epochs}
    sklearn_perceptron_params = {"eta0": learning_rate, "max_iter": epochs}

    # Load the dataset and create training and testing sets
    X, y = preprocessing.load_data("data/breast_cancer_wisconsin_diagnostic/wdbc.data", 1, list(range(2, 32)))
    
    X_train, y_train, X_test, y_test = preprocessing.split_data(X, y)

    # Evaluate custom and scikit-learn KNN implementations
    custom_knn_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=knn.KNN,
        model_params=custom_knn_params,
        scaler_class=custom_knn_scaler
    )

    sklearn_knn_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=KNeighborsClassifier,
        model_params=sklearn_knn_params,
        scaler_class=sklearn_knn_scaler
    )

    # Evaluate custom and scikit-learn Logistic Regression implementations
    custom_log_reg_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=logistic_regression.Logistic_Regression,
        model_params=custom_log_reg_params,
        scaler_class=custom_log_reg_scaler
    )

    sklearn_log_reg_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=LogisticRegression,
        model_params=sklearn_log_reg_params,
        scaler_class=sklearn_log_reg_scaler
    )

    # Evaluate custom and scikit-learn Perceptron implementations
    custom_perceptron_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=perceptron.Perceptron,
        model_params=custom_perceptron_params,
        scaler_class=custom_perceptron_scaler
    )

    sklearn_perceptron_results = evaluation.cross_validation(
        X_train=X_train,
        y_train=y_train,
        num_folds=num_folds,
        algorithm=Perceptron,
        model_params=sklearn_perceptron_params,
        scaler_class=sklearn_perceptron_scaler
    )

    # Print evaluation results for all models
    print_results("Custom KNN", custom_knn_results, num_folds)
    print_results("Sklearn KNN", sklearn_knn_results, num_folds)

    print_results("Custom Logistic Regression", custom_log_reg_results, num_folds)
    print_results("Sklearn Logistic Regression", sklearn_log_reg_results, num_folds)

    print_results("Custom Perceptron", custom_perceptron_results, num_folds)
    print_results("Sklearn Perceptron", sklearn_perceptron_results, num_folds)


if __name__ == "__main__":
    main()