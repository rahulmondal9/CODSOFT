# Credit Card Fraud Detection

A machine learning project that classifies credit card transactions as **genuine (Class 0)** or **fraudulent (Class 1)** using a Logistic Regression model. The notebook explores transaction data, checks for missing values, scales the transaction amount, balances the classes with SMOTE, trains a classifier, and evaluates the predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

> **Recorded notebook result:** 97.48% accuracy on 113,726 evaluation rows. Important caveat: the notebook applies scaling and SMOTE before the train/test split, which causes data leakage and can make the reported evaluation overly optimistic. Treat this score as preliminary until the workflow is corrected and rerun.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Class Distribution](#class-distribution)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Evaluation Metrics](#evaluation-metrics)
- [Accuracy and Test Results](#accuracy-and-test-results)
- [Efficiency and Performance](#efficiency-and-performance)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Recommended Improvements](#limitations-and-recommended-improvements)
- [Conclusion](#conclusion)

## Project Overview

Credit card fraud detection is a binary classification problem: a system examines transaction attributes and estimates whether a transaction is legitimate or fraudulent. Because fraudulent transactions are rare compared with genuine transactions, this task is challenging and requires careful evaluation beyond accuracy alone.

This notebook uses a Logistic Regression classifier from Scikit-learn. It loads transaction data, inspects the class distribution and missing values, standardizes the `Amount` feature, applies Synthetic Minority Over-sampling Technique (SMOTE), trains the model, and reports classification metrics.

The project is a learning-oriented baseline. The notebook does not implement a live payment gateway, real-time transaction stream, alerting service, or production deployment.

## Objectives

- Explore a labelled credit card transaction dataset.
- Understand the imbalance between genuine and fraudulent transactions.
- Prepare transaction features for machine learning.
- Apply SMOTE to address minority-class imbalance.
- Train a Logistic Regression classifier.
- Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
- Identify limitations and improvements needed for a trustworthy fraud-detection evaluation.

## Dataset

The notebook reads a CSV dataset from a ZIP archive using Pandas. The local file path in the original notebook is specific to one Windows computer and must be updated for another environment.

The loaded dataset contains **284,807 rows and 31 columns**.

| Field | Description |
|---|---|
| `Time` | Time elapsed, in seconds, from the first transaction in the dataset |
| `V1`–`V28` | Transformed/anonymized numerical features supplied with the dataset |
| `Amount` | Transaction amount |
| `Class` | Target label: `0` for genuine transactions and `1` for fraud |

The notebook does not explain the original meanings of `V1`–`V28`; they should be treated as supplied transformed features rather than interpreted as specific customer or transaction attributes.

## Class Distribution

The initial dataset contains:

| Class | Meaning | Transactions | Approximate share |
|---|---|---:|---:|
| 0 | Genuine | 284,315 | 99.827% |
| 1 | Fraud | 492 | 0.173% |
| **Total** | | **284,807** | **100%** |

This is a highly imbalanced dataset: fraud cases make up only about 0.17% of the original rows. In this situation, a model can obtain a deceptively high accuracy by predicting almost every transaction as genuine. Therefore, fraud-class recall, precision, F1-score, the confusion matrix, and precision-recall AUC are important metrics to evaluate.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Loading and inspecting transaction data |
| NumPy | Numerical data handling |
| Matplotlib | Plotting |
| Seaborn | Distribution plots and confusion matrix visualization |
| Scikit-learn | Scaling, data splitting, Logistic Regression, and evaluation metrics |
| imbalanced-learn | SMOTE oversampling |

## Workflow

1. **Load the dataset:** read transaction data from the supplied ZIP archive.
2. **Inspect the data:** examine the first rows, dataset shape, and missing values.
3. **Explore class distribution:** count genuine and fraudulent transactions and visualize selected distributions.
4. **Scale transaction amount:** standardize the `Amount` column.
5. **Separate features and target:** use all columns except `Class` as input features and `Class` as the target.
6. **Apply SMOTE:** synthesize minority-class examples to balance the dataset.
7. **Split the data:** create training and test partitions.
8. **Train Logistic Regression:** fit the classifier to the training data.
9. **Generate predictions:** predict class labels and fraud probabilities.
10. **Evaluate:** calculate accuracy, precision, recall, F1-score, and plot a confusion matrix.

**Important methodological note:** the current notebook scales `Amount` and applies SMOTE before splitting the data. For a reliable evaluation, the split should happen first, the scaler should be fitted only on training data, and SMOTE should be applied only to the training partition.

## Data Preprocessing

### Missing-value check

The notebook checks missing values in every column. Its recorded output reports **zero missing values** across the 31 columns, so no missing-value imputation is performed.

### Feature scaling

The notebook applies `StandardScaler` to the `Amount` column:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
```

Standardization transforms values based on the feature's mean and standard deviation. This is useful because Logistic Regression is sensitive to feature scale.

However, the scaler is fitted on the entire dataset before splitting. This allows information from the future test partition to influence preprocessing. Fit the scaler on training data only, then transform validation and test data.

### Feature and target separation

The notebook uses all columns other than `Class` as features:

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

The target column `Class` contains the binary label that the model learns to predict.

### Handling class imbalance with SMOTE

The notebook uses SMOTE with `random_state=42`:

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
```

SMOTE creates synthetic minority-class examples by interpolating between minority-class samples. This makes the classes more balanced for training.

In the current notebook, SMOTE is applied **before** the train/test split. Consequently, synthetic samples can be generated using observations that later appear in the test partition, leading to leakage and an overly optimistic evaluation. SMOTE must be applied to training data only, after splitting.

## Model

The notebook trains a **Logistic Regression** classifier using Scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
```

Logistic Regression estimates the probability of a transaction belonging to the fraud class and classifies it using a decision threshold, 0.5 by default.

### Why Logistic Regression?

- It is a relatively simple and interpretable baseline for binary classification.
- It can produce probability estimates for fraud-risk ranking.
- It trains and predicts efficiently on many tabular datasets.
- Its coefficients can help explain how input features relate to the predicted class, subject to scaling and feature interpretation.

### Convergence warning

The saved notebook output includes a `ConvergenceWarning`: the `lbfgs` solver reached its iteration limit before convergence.

This means the optimizer may not have fully converged to its solution. The reported metrics were produced despite the warning, but the model should be retrained with an appropriate preprocessing pipeline and a higher `max_iter` value, then checked for convergence.

For example:

```python
model = LogisticRegression(
    random_state=42,
    max_iter=2000
)
```

Increasing `max_iter` is a practical starting point, not a guarantee that all modeling or evaluation issues are resolved.

## Evaluation Metrics

The notebook reports accuracy, precision, recall, and F1-score through Scikit-learn's `classification_report`, and visualizes a confusion matrix.

- **Accuracy:** the fraction of all transactions classified correctly.
- **Precision (fraud class):** the fraction of transactions predicted as fraud that are actually fraud.
- **Recall (fraud class):** the fraction of actual fraud transactions detected by the model.
- **F1-score:** the harmonic mean of precision and recall.
- **Confusion matrix:** shows true negatives, false positives, false negatives, and true positives.

In fraud detection, false negatives are especially important because they represent fraudulent transactions missed by the model. False positives also matter because they may cause genuine transactions to be flagged or declined.

For a more complete evaluation, add **average precision / PR-AUC**, ROC-AUC, precision and recall at a chosen operational threshold, and an evaluation on the original imbalanced test distribution.

## Accuracy and Test Results

The saved notebook reports the following results after applying SMOTE and splitting the resampled data:

| Metric | Class 0 (Genuine) | Class 1 (Fraud) |
|---|---:|---:|
| Precision | 0.97 | 0.98 |
| Recall | 0.98 | 0.97 |
| F1-score | 0.97 | 0.97 |
| Support | 56,750 | 56,976 |

| Overall metric | Recorded result |
|---|---:|
| Accuracy | **97.48%** |
| Evaluation rows | **113,726** |
| Macro-average F1-score | 0.97 |
| Weighted-average F1-score | 0.97 |

The evaluation set is nearly balanced because SMOTE was applied before splitting. These metrics therefore do not represent performance on the original transaction distribution, where only 492 of 284,807 transactions are fraudulent.

The 97.48% accuracy is the result recorded in the notebook, but it should not be considered a trustworthy estimate of real-world fraud detection until the preprocessing and sampling leakage is corrected. The notebook does not record a reliable final evaluation on a separate, untouched test set with the original class distribution.

### Confusion matrix

The notebook plots a confusion matrix with the labels `Normal` and `Fraud`. It does not print the matrix values as text in the saved output. After fixing the evaluation protocol, report the actual counts of true negatives, false positives, false negatives, and true positives to make the results easier to audit.

## Efficiency and Performance

### Computational characteristics

Logistic Regression is generally less computationally expensive than many complex ensemble or deep learning models, and it supports probability predictions. However, the notebook does not record a controlled training-time benchmark, inference latency, memory usage, or deployment throughput. Therefore, no precise speed or production-efficiency claim can be made from this notebook alone.

### Observed model issue

The training output includes a convergence warning. Retraining with a suitable iteration limit and properly scaled features is recommended before comparing performance.

### Practical efficiency improvements

- Use a Scikit-learn `Pipeline` and `imblearn.pipeline.Pipeline` to keep preprocessing and resampling organized.
- Split the data before fitting scalers or applying SMOTE.
- Apply SMOTE only to the training partition.
- Tune Logistic Regression parameters using cross-validation on the training data.
- Measure training time and per-transaction prediction latency under a documented environment.
- Compare Logistic Regression with models such as Random Forest or gradient-boosted trees using the same leakage-free split and metrics.
- Choose a probability threshold based on operational costs rather than assuming 0.5 is always appropriate.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- The credit card fraud dataset used by the notebook.
- The required Python packages.

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### Dataset path

The notebook uses an absolute Windows path to `archive.zip`. Replace it with the path to your local dataset, or use a relative path such as:

```python
df = pd.read_csv("data/archive.zip")
```

Ensure the archive contains the expected CSV file and that the path matches your repository layout. Do not commit sensitive financial data or large datasets to a public repository unless you have the right to distribute them.

## How to Run

1. Clone or download the repository.
2. Download the dataset from its authorized source and place it in the expected location.
3. Install the dependencies.
4. Update the dataset path in the notebook.
5. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Credit_Card_Fraud_Detection.ipynb`.
7. Run the notebook cells in order.
8. Review the class distribution, training warning, classification report, and confusion matrix.
9. Before relying on the results, fix the data leakage issues and rerun the evaluation.

## Project Structure

A recommended repository structure is:

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── requirements.txt
└── data/
    └── archive.zip   # Keep local; do not commit unless permitted
```

- `Credit_Card_Fraud_Detection.ipynb` — notebook for data exploration, preprocessing, training, and evaluation.
- `README.md` — project documentation.
- `requirements.txt` — Python dependencies.
- `data/` — optional local dataset directory.

The `data/` folder and dataset file are suggested structure examples; they are not created by the notebook itself.

## Limitations and Recommended Improvements

### 1. Prevent data leakage

The notebook applies `StandardScaler` and SMOTE before splitting into train and test sets. This allows information from the evaluation partition to influence preprocessing and synthetic sample generation.

**Recommended fix:** split the original data first. Fit the scaler on training data only, and run SMOTE only on the training partition. A pipeline can help enforce this order.

### 2. Evaluate on the original class distribution

The notebook evaluates on a resampled, nearly balanced test set rather than the naturally imbalanced transaction distribution.

**Recommended fix:** keep the test set untouched and at its original class distribution. Use SMOTE only for training and report fraud precision, fraud recall, F1-score, PR-AUC, and confusion-matrix counts on the untouched test set.

### 3. Resolve the convergence warning

The Logistic Regression solver reaches its iteration limit.

**Recommended fix:** use a consistent preprocessing pipeline, scale numerical features appropriately, increase `max_iter`, and confirm that the solver converges.

### 4. Tune the decision threshold

The default probability threshold may not reflect the cost of missed fraud versus incorrectly flagged genuine transactions.

**Recommended fix:** choose a threshold using validation data and a clear operational objective, then report its effect on precision, recall, and false-positive rate.

### 5. Improve model comparison

The notebook trains one Logistic Regression model and does not include a systematic model comparison or cross-validation.

**Recommended fix:** compare multiple models using the same leakage-free splits and metrics, and use cross-validation only within the training data.

### 6. Consider time-aware evaluation

Transactions are time-dependent, and random splits may not represent future deployment conditions.

**Recommended fix:** where appropriate, evaluate on a later time period than the training period and document the split strategy. This better reflects how a model may perform on future transactions.

### 7. Avoid unsupported deployment claims

The notebook is an offline experiment. It does not include a production API, streaming ingestion, monitoring, retraining, or a payment-system integration.

**Recommended fix:** treat deployment as a separate engineering stage involving security, latency, monitoring, alerting, and ongoing model evaluation.

## Conclusion

This project demonstrates a baseline credit card fraud detection workflow using Pandas, Scikit-learn, Logistic Regression, and SMOTE. The notebook explores **284,807 transactions**, including **492 fraud cases**, and records **97.48% accuracy** on a resampled evaluation set containing 113,726 rows.

The result is preliminary because SMOTE and feature scaling are performed before the train/test split, and the Logistic Regression training run produces a convergence warning. The most important next step is to correct the data pipeline, evaluate on an untouched test set with the original class imbalance, and report fraud-focused metrics such as precision, recall, F1-score, and PR-AUC.

---

**Reproducibility note:** The metrics in this README are taken from the saved notebook output. They may change after correcting the evaluation pipeline, updating dependencies, or retraining the model.
