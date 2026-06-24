# K-Nearest Neighbor Classification – Breast Cancer Diagnostic Tool

## Project Overview
This repository contains a machine learning project demonstrating the implementation of a K-Nearest Neighbor (KNN) classifier using Python and `scikit-learn`. The model analyzes the Breast Cancer Wisconsin dataset to predict whether a tumor is malignant or benign based on its computed features.

The analysis covers exploratory data visualization, the critical necessity of feature scaling (StandardScaler) for distance-based algorithms, hyperparameter tuning via cross-validation, and comprehensive model evaluation using confusion matrices, precision, and recall.

## Dataset Description
The dataset is loaded natively via `sklearn.datasets.load_breast_cancer()`. It contains 569 instances with 30 numeric, predictive attributes (e.g., radius, texture, perimeter, area, smoothness). 
* **Target:** 0 (Malignant), 1 (Benign)

## Setup Instructions
1. Clone this repository to your local machine.
2. Install the required dependencies using the terminal:
   `pip install -r requirements.txt`
3. Open `knn_classification.ipynb` in VS Code or JupyterLab.
4. Run all cells to view the feature distributions, the cross-validation optimization curve, and the final classification metrics.

## Key Findings & Evaluation
* **Feature Scaling is Mandatory:** Because KNN utilizes Euclidean distance calculations, unscaled data heavily biases the model toward features with larger magnitudes (like area). `StandardScaler` successfully normalized the weights.
* **Optimal 'k':** By plotting cross-validation accuracy against various $k$ values (1, 3, 5, 7, 9, 11), we determined the optimal number of neighbors to prevent overfitting while maintaining high predictive power.
* **Model Performance:** The final model achieved exceptional Accuracy and Recall. In a medical diagnostic context, a high Recall for malignant tumors is critical, as it drastically reduces the number of dangerous False Negatives.