
import matplotlib.pyplot as plt
import numpy as np


def create_plot(name: str, dataset_name: str, hyperparams: dict, *metrics: dict) -> None:
    """
    Creates a bar chart comparing custom and Scikit-learn implementations
    of an algorithm using accuracy, precision, recall, and F1 scores.

    Args:
        name: Name of the algorithm being compared.
        dataset_name: Name of the dataset used for evaluation.
        hyperparams: Dictionary containing the model's hyperparameters.
        *metrics: Two dictionaries containing the custom and Scikit-learn
                  evaluation results, respectively.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    custom_metrics = [metrics[0][key][0] * 100 for key in ["accuracy", "precision", "recall", "f1"]]
    sklearn_metrics = [metrics[1][key][0] * 100 for key in ["accuracy", "precision", "recall", "f1"]]

    x = np.arange(len(metric_names))
    width = 0.35
    hyperparam_text = ", ".join(f"{key}={value}" for key, value in hyperparams.items())

    plt.figure(figsize=(9, 6))

    custom_bars = plt.bar(x - width/2, custom_metrics, width,
                          color="#4C78A8", label=f"Custom {name}")
    sklearn_bars = plt.bar(x + width/2, sklearn_metrics, width,
                           color="#F58518", label=f"Scikit-learn {name}")

    plt.bar_label(custom_bars, fmt="%.2f%%", padding=3)
    plt.bar_label(sklearn_bars, fmt="%.2f%%", padding=3)

    plt.suptitle(f"Custom vs. Scikit-learn {name} Implementation",
                 fontsize=14, fontweight="bold")
    plt.title(f"Dataset: {dataset_name} | {hyperparam_text}", fontsize=10)

    plt.xticks(x, metric_names, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 110)

    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.gca().set_axisbelow(True)

    plt.ylabel("Score (%)", fontsize=11)
    plt.xlabel("Metrics", fontsize=11)

    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
