
# ==================================================
# Imports
# ==================================================

import preprocessing
import evaluation
import visualization

import algorithms.knn as knn
import algorithms.logistic_regression as logistic_regression
import algorithms.perceptron as perceptron

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==================================================
# Result Display
# ==================================================

def print_results(model_name: str, results: dict, num_folds: int = 1) -> None:
    """
    Prints the evaluation results for a classification model.

    Displays accuracy, precision, recall, and F1-score for each
    fold and their averages, or the test set results, along with
    the combined confusion matrix.

    Args:
        model_name: Name of the model being evaluated.
        results: Dictionary containing evaluation metrics and confusion matrix.
        num_folds: Number of cross-validation folds.
    """
    print(f"\n{'=' * 50}")
    print(f"{model_name:^50}")
    print(f"{'=' * 50}")

    print(f"{'Metric':<15}", end="")

    if num_folds > 1:
        for i in range(1, num_folds + 1):
            print(f"{'Fold ' + str(i):>8}", end="")
        print(f"{'Average':>10}")
    else:
        print(f"{'Test Set':>8}")
        print()

    print("-" * 65)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = results[metric]
        print(f"{metric.capitalize():<15}", end="")

        for score in scores[:-1]:
            print(f"{score * 100:>7.2f}%", end="")

        print(f"{scores[-1] * 100:>9.2f}%")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


# ==================================================
# Main Experiment
# ==================================================

def main():
    """
    Runs the machine learning experiments.

    Loads and splits the dataset, configures custom and scikit-learn
    models, evaluates each model, and visualizes the resulting metrics.
    """

    # --------------------------------------------------
    # Experiment Settings
    # --------------------------------------------------

    num_folds = 10

    # Scalers
    custom_standard_scaler = preprocessing.CustomStandardScaler
    custom_minmax_scaler = preprocessing.CustomMinMaxScaler

    sklearn_standard_scaler = StandardScaler
    sklearn_minmax_scaler = MinMaxScaler

    # --------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------

    custom_knn_params = {"k": 7}
    sklearn_knn_params = {"n_neighbors": 7}

    custom_log_reg_params = {
        "learning_rate": 0.8,
        "epochs": 100,
        "lambda_reg": 0.01
    }
    sklearn_log_reg_params = {"max_iter": 100}

    custom_perceptron_params = {
        "learning_rate": 0.1,
        "epochs": 100
    }
    sklearn_perceptron_params = {
        "eta0": 0.1,
        "max_iter": 100,
        "random_state": 15
    }

    # ------------------------------- # Model Configurations
    # --------------------------------------------------

    models = [
        ("Custom KNN", knn.KNN, custom_knn_params, custom_standard_scaler),
        ("Sklearn KNN", KNeighborsClassifier, sklearn_knn_params, sklearn_standard_scaler),

        ("Custom Logistic Regression", logistic_regression.Logistic_Regression,
         custom_log_reg_params, custom_standard_scaler),
        ("Sklearn Logistic Regression", LogisticRegression,
         sklearn_log_reg_params, sklearn_standard_scaler),

        ("Custom Perceptron", perceptron.Perceptron,
         custom_perceptron_params, custom_standard_scaler),
        ("Sklearn Perceptron", Perceptron,
         sklearn_perceptron_params, sklearn_standard_scaler)
    ]

    # --------------------------------------------------
    # Dataset Loading
    # --------------------------------------------------

    # Student Performance Dataset
    x_cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
        "health", "absences", "G1", "G2"
    ]

    X, y = preprocessing.load_data(
        dataset="data/student_performance/student-mat.csv",
        y_col="G3",
        x_cols=x_cols,
        header=0
    )

    # Breast Cancer Wisconsin Diagnostic Dataset
    # X, y = preprocessing.load_data(
    #     dataset="data/breast_cancer_wisconsin_diagnostic/wdbc.data",
    #     y_col=1,
    #     x_cols=list(range(2, 32))
    # )

    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    X_train, y_train, X_test, y_test = preprocessing.split_data(X, y)

    # --------------------------------------------------
    # Cross-Validation
    # --------------------------------------------------

    # print("\nCROSS-VALIDATION RESULTS")
    # cv_results = {}

    # for name, algorithm, params, scaler in models:
    #     results = evaluation.cross_validation(
    #         X_train=X_train,
    #         y_train=y_train,
    #         num_folds=num_folds,
    #         algorithm=algorithm,
    #         model_params=params,
    #         scaler_class=scaler
    #     )
    #
    #     cv_results[name] = results
    #     print_results(name, results, num_folds)

    # --------------------------------------------------
    # Held-Out Test Set Evaluation
    # --------------------------------------------------

    print("\nTEST SET RESULTS")

    test_results = {}

    for name, algorithm, params, scaler in models:
        results = evaluation.evaluate_test(
            X_test=X_test,
            X_train=X_train,
            y_test=y_test,
            y_train=y_train,
            algorithm=algorithm,
            model_params=params,
            scaler_class=scaler
        )

        test_results[name] = results
        print_results(name, results)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    plots = [
        ("KNN", custom_knn_params, "Custom KNN", "Sklearn KNN"),
        ("Logistic Regression", custom_log_reg_params,
         "Custom Logistic Regression", "Sklearn Logistic Regression"),
        ("Perceptron", custom_perceptron_params,
         "Custom Perceptron", "Sklearn Perceptron")
    ]

    for name, params, custom_name, sklearn_name in plots:
        visualization.create_plot(
            name,
            "student_performance",
            params,
            test_results[custom_name],
            test_results[sklearn_name]
        )

    plt.show()


if __name__ == "__main__":
    main()
