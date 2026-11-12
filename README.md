# Machine Learning Algorithm Comparison

## Overview

This project implements and compares three supervised machine learning classification algorithms: K-Nearest Neighbors (KNN), Logistic Regression, and Perceptron.

The custom algorithms were developed from scratch using NumPy and are compared against their corresponding Scikit-learn implementations. The models are evaluated across two datasets to compare their performance on different classification tasks.

## Results

The following results represent performance on the held-out test sets.

### Breast Cancer Wisconsin Diagnostic (WDBC)

| Algorithm | Custom Accuracy | Scikit-learn Accuracy |
|---|---:|---:|
| KNN | 97.37% | 97.37% |
| Logistic Regression | 98.25% | 99.12% |
| Perceptron | 97.37% | 97.37% |

### Student Performance

| Algorithm | Custom Accuracy | Scikit-learn Accuracy |
|---|---:|---:|
| KNN | 48.10% | 48.10% |
| Logistic Regression | 72.15% | 73.42% |
| Perceptron | 75.95% | 62.03% |

### Key Findings

- **KNN:** Both implementations achieved identical accuracy on both datasets.
- **Logistic Regression:** Scikit-learn performed slightly better on both datasets.
- **Perceptron:** Both implementations achieved identical accuracy on WDBC. On Student Performance, the custom implementation outperformed Scikit-learn by 13.92%
- **Dataset Differences:** All three algorithms performed better on WDBC than on Student Performance, demonstrating how model performance varies across datasets and classification tasks.

## Visualizations

Comparison plots show custom and Scikit-learn performance across accuracy, precision, recall, and F1 score.

Plots are available in the `plots/` directory.
[View all plots](plots/)
## Algorithms

The following algorithms were implemented from scratch using NumPy:

- **K-Nearest Neighbors (KNN):** Classifies samples based on the labels of their nearest neighbors.
- **Logistic Regression:** Uses sigmoid activation for binary classification and softmax for multiclass classification.
- **Perceptron:** Implements a linear classification algorithm for binary and multiclass classification.

Each custom implementation is compared against its corresponding Scikit-learn implementation.

## Datasets

### Breast Cancer Wisconsin Diagnostic (WDBC)
- Binary classification dataset containing 30 numerical features.
- Predicts whether a breast cancer diagnosis is malignant or benign.
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Student Performance
- Uses the Mathematics dataset from the UCI Student Performance dataset.
- Predicts student performance using four categories based on final grades (G3):
  - Class 0: 0–9
  - Class 1: 10–13
  - Class 2: 14–16
  - Class 3: 17–20
- Includes numerical, binary categorical, and one-hot encoded categorical features.
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

## Implementation

The project includes:
- Data loading and preprocessing
- Categorical feature encoding
- 80/20 training and testing split
- Custom StandardScaler and MinMaxScaler implementations
- Model training and prediction
- Stratified k-fold cross-validation
- Held-out test set evaluation
- Accuracy, weighted precision, weighted recall, and weighted F1 score
- Confusion matrices

### Feature Scaling

Both custom and Scikit-learn scaling implementations were used:

- **StandardScaler:** Used for all algorithm and dataset combinations except WDBC KNN.
- **MinMaxScaler:** Used for both custom and Scikit-learn KNN implementations on WDBC.

Scalers are fitted on training data and then applied to validation or test data to prevent data leakage.

## Limitations

- Only the Mathematics dataset (`student-mat.csv`) from the UCI Student Performance dataset was evaluated. The Mathematics and Portuguese datasets were not combined, and the Portuguese dataset was not included.
- Model performance depends on the selected hyperparameters and preprocessing methods. The results represent the configurations used in these experiments.
- Test results are based on a single 80/20 train-test split.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Running the Project

1. Clone the repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main file:

   ```bash
   python main.py
   ```
