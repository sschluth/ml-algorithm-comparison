import preprocessing
import evaluation
import knn
import logistic_regression
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    X, y = preprocessing.load_data("data/breast_cancer_wisconsin_diagnostic/wdbc.data", 1, list(range(2, 32)))
    
    X_train, y_train, X_test, y_test = preprocessing.split_data(X, y)


    knn_accuracies = evaluation.cross_validation(
        X_train,
        y_train,
        5,
        knn.KNN,
        {"k": 5},
        preprocessing.CustomMinMaxScaler
    )

    sk_knn_accuracies = evaluation.cross_validation(
        X_train,
        y_train,
        5,
        KNeighborsClassifier,
        {"n_neighbors": 5},
        MinMaxScaler
    )

    log_reg_accuracies = evaluation.cross_validation(
        X_train,
        y_train,
        5,
        logistic_regression.Logistic_Regression,
        {"learning_rate": 0.1, "epoch": 5},
        preprocessing.CustomStandardScaler
    )

    sk_log_reg_accuracies = evaluation.cross_validation(
        X_train,
        y_train,
        5,
        LogisticRegression,
        {"max_iter": 5},
        StandardScaler
    )

    print("KNN Cross-Validation Results")
    print("----------------------------")

    print(f"Custom KNN:     {knn_accuracies[-1]:.4f}")
    print(f"Sklearn KNN:    {sk_knn_accuracies[-1]:.4f}")

    print("\nIndividual Fold Accuracies:")
    for i, (custom, sklearn) in enumerate(zip(knn_accuracies[:-1], sk_knn_accuracies[:-1])):
        print(f"Fold {i + 1}: Custom = {custom:.4f} | Sklearn = {sklearn:.4f}")


    print("\nLogistic Regression Cross-Validation Results")
    print("----------------------------")

    print(f"Custom Logistic Regression:     {log_reg_accuracies[-1]:.4f}")
    print(f"Sklearn Logistic Regression:    {sk_log_reg_accuracies[-1]:.4f}")

    print("\nIndividual Fold Accuracies:")
    for i, (custom, sklearn) in enumerate(zip(log_reg_accuracies[:-1], sk_log_reg_accuracies[:-1])):
        print(f"Fold {i + 1}: Custom = {custom:.4f} | Sklearn = {sklearn:.4f}")

if __name__ == "__main__":
    main()