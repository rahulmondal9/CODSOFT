# Iris Flower Classification Using Machine Learning

A machine learning project that explores the Iris flower dataset, visualizes flower measurements, groups observations with K-Means clustering, and classifies Iris species using a Random Forest model. The notebook uses sepal and petal measurements, produces exploratory visualizations, and evaluates classification with accuracy, a classification report, and a confusion matrix.

> **Recorded notebook result:** 100.00% accuracy on a 30-row test set. This is the result saved in the notebook, not a guarantee of perfect performance on new data. The notebook also creates a K-Means cluster feature using the full dataset before the train/test split, so the evaluation should be rerun with a leakage-free preprocessing workflow before this score is treated as a robust benchmark.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset and Features](#dataset-and-features)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [K-Means Clustering](#k-means-clustering)
- [Preprocessing](#preprocessing)
- [Random Forest Classifier](#random-forest-classifier)
- [Accuracy and Test Results](#accuracy-and-test-results)
- [Efficiency and Performance](#efficiency-and-performance)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Improvements](#limitations-and-improvements)
- [Conclusion](#conclusion)

## Project Overview

The Iris flower classification problem uses measurements of flower sepals and petals to identify the species. This notebook explores the measurements with plots, clusters the observations into three groups using K-Means, and trains a Random Forest classifier to predict the species label.

The four measurement attributes are:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target column is `species`. The notebook also creates a `cluster` column using K-Means and includes this column in the features passed to the Random Forest model.

The notebook demonstrates both **unsupervised learning** (K-Means clustering) and **supervised learning** (Random Forest classification).

## Objectives

- Load and inspect the Iris dataset.
- Check the dataset for missing values.
- Explore the relationship between flower measurements and species using visualizations.
- Group flowers into three clusters using K-Means.
- Apply Min-Max scaling as a preprocessing demonstration.
- Train a Random Forest classifier to predict species.
- Evaluate predictions using accuracy, a classification report, and a confusion matrix.

## Dataset and Features

The notebook reads a CSV file from a local ZIP archive with Pandas. The original notebook uses an absolute Windows file path, which must be updated for another computer.

The saved output shows the following columns:

| Column | Description |
|---|---|
| `sepal_length` | Length of the sepal |
| `sepal_width` | Width of the sepal |
| `petal_length` | Length of the petal |
| `petal_width` | Width of the petal |
| `species` | Flower species label |

The displayed data contains species names such as `Iris-setosa`. The Random Forest workflow label-encodes the species names into integers for training.

The notebook does not print the full dataset shape in its saved outputs. The classification report shows 30 samples in the held-out test set.

## Technologies Used

| Library / Tool | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading and inspecting the dataset |
| Scikit-learn | K-Means, MinMaxScaler, train/test split, LabelEncoder, Random Forest, and metrics |
| Matplotlib | Scatter plots and charts |
| Seaborn | Violin plots, pair plots, and confusion-matrix visualization |
| Jupyter Notebook | Interactive development and analysis |

## Project Workflow

1. **Load data:** read the Iris dataset from a ZIP archive.
2. **Inspect missing values:** count null values in each column.
3. **Explore distributions:** plot violin charts and pairwise relationships by species.
4. **Apply K-Means:** assign each flower to one of three clusters based on its four measurements.
5. **Visualize clusters:** plot sepal and petal measurements colored by cluster.
6. **Demonstrate scaling:** use `MinMaxScaler` to scale the four measurement columns in a temporary dataframe.
7. **Prepare supervised learning data:** separate measurements and the target species label.
8. **Encode labels:** convert text species labels to integer classes.
9. **Split data:** use an 80/20 train/test split with `random_state=42`.
10. **Train a classifier:** fit a Random Forest model.
11. **Evaluate:** calculate accuracy, print a classification report, and plot a confusion matrix.

## Exploratory Data Analysis

The notebook uses the following visualizations to explore the dataset.

### Violin Plots

Violin plots compare the distribution of each measurement—sepal length, sepal width, petal length, and petal width—across the species. They can reveal differences in central tendency, spread, and distribution shape.

### Pair Plot

A Seaborn pair plot shows pairwise relationships between measurements and colors observations by species. It helps identify features that visually separate different species.

### K-Means Scatter Plots

The notebook creates scatter plots for:

- Sepal length versus sepal width.
- Petal length versus petal width.

Observations are colored by their assigned K-Means cluster. These plots help compare cluster structure across sepal and petal measurements.

## K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions observations into a specified number of clusters. The notebook initializes it with three clusters:

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)
var1 = km.fit_predict(
    df[[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]]
)
df1["cluster"] = var1
```

Each observation receives a cluster ID: `0`, `1`, or `2`. These values are arbitrary cluster identifiers, not the original species labels. A cluster ID should not automatically be interpreted as a particular Iris species.

The notebook displays the clusters in scatter plots using the sepal and petal measurements.

**Reproducibility note:** the K-Means calls do not set `random_state`, and the saved notebook shows a scikit-learn warning about the default `n_init` value changing in a future release. For repeatable clustering, set both options explicitly, for example `KMeans(n_clusters=3, random_state=42, n_init=10)`.

## Preprocessing

### Missing-Value Check

The notebook runs:

```python
df.isnull().sum()
```

The saved output shows zero missing values in the four measurement columns and the species column. No missing-value imputation is carried out.

### Min-Max Scaling

The notebook demonstrates `MinMaxScaler` on the four numerical measurement columns. This transformation maps each feature to the range from 0 to 1 based on its minimum and maximum.

However, after scaling `df1`, a later cell resets it with `df1 = df.copy()` before K-Means is run again. Therefore, the supervised Random Forest stage uses the original, unscaled measurements rather than the scaled dataframe.

Random Forest models generally do not require feature scaling, so using the raw measurement features is acceptable for that model. The key point is that the scaling step shown in the notebook is not retained in the final classifier's input data.

### Label Encoding and Feature Selection

The notebook encodes the string species labels using `LabelEncoder`, then separates the target and features:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df1["species"])

X = df1.drop(columns=["species"])
y = df1["species"]
```

**Important:** `X` contains every column except `species`, including the generated `cluster` column. Consequently, the Random Forest is trained using the four numerical measurements plus the cluster ID.

## Random Forest Classifier

The notebook trains a `RandomForestClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

A Random Forest combines predictions from multiple decision trees. Each tree learns decision rules from features, and the forest combines the trees' predictions to classify an observation.

The notebook uses the Scikit-learn default configuration except for `random_state=42`. In common Scikit-learn versions, the default forest uses 100 trees; because the notebook does not explicitly set `n_estimators`, the exact effective defaults depend on the installed version.

### Train/Test Split

The notebook uses:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

This reserves 20% of the rows for evaluation and uses a fixed split seed. The call does not specify `stratify=y_encoded`, so the class proportions in the test set are not explicitly preserved.

## Accuracy and Test Results

The saved notebook output reports:

| Metric | Recorded Result |
|---|---:|
| Test accuracy | **100.00%** |
| Test examples | **30** |
| Correctly classified examples implied by the report | **30 of 30** |
| Incorrectly classified examples implied by the report | **0 of 30** |

The classification report prints the following metrics:

| Encoded class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 1.00 | 1.00 | 10 |
| 1 | 1.00 | 1.00 | 1.00 | 9 |
| 2 | 1.00 | 1.00 | 1.00 | 11 |
| **Overall accuracy** | | | **1.00** | **30** |
| Macro average | 1.00 | 1.00 | 1.00 | 30 |
| Weighted average | 1.00 | 1.00 | 1.00 | 30 |

The report is printed with encoded class IDs (`0`, `1`, `2`) rather than the original species names. To make the output easier to interpret, pass `target_names=label_encoder.classes_` to `classification_report`, ensuring the encoder and report use the same class order.

### Confusion Matrix

The notebook computes a confusion matrix from `y_test` and `y_pred`, then plots it using Seaborn. Given the saved classification report's 100% precision and recall for all three classes, the reported matrix should contain only diagonal counts, corresponding to 10, 9, and 11 test examples per encoded class.

This is a result from the current split, not evidence that the classifier will be perfect on all future flower measurements.

## Efficiency and Performance

### Computational Characteristics

Random Forest is a practical baseline for small tabular datasets. It learns nonlinear decision boundaries and does not require standardized inputs. The Iris dataset is small, and the notebook does not show a need for GPU acceleration.

### What the Notebook Measures

The notebook records classification metrics but does not record:

- Model fitting time.
- Per-sample inference latency.
- Peak memory usage.
- Saved model size.
- Repeated cross-validation scores.

Therefore, no exact runtime or production-efficiency claim can be made from this notebook's outputs.

### Practical Efficiency Improvements

- Record the training and prediction durations with `time.perf_counter()`.
- Set `n_estimators`, `max_depth`, and other model parameters explicitly to make configuration clear.
- Compare Random Forest against simple baselines such as Logistic Regression and Decision Tree.
- Use cross-validation for a more stable estimate than one 30-sample test split.
- Keep preprocessing and feature creation inside a reproducible pipeline.

## Installation

### Prerequisites

- Python 3.10 or another compatible Python version.
- Jupyter Notebook or JupyterLab.
- The libraries listed below.
- The Iris CSV archive used by the notebook.

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

### Dataset Path

The notebook reads data from an absolute Windows path. Change the path for your environment, for example:

```python
import pandas as pd

df = pd.read_csv("data/iris.csv")
```

Use the correct path and file name for the dataset you have. The relative path above is an example; the notebook itself does not create the `data/iris.csv` file.

## How to Run

1. Clone or download the repository.
2. Ensure the Iris dataset archive or CSV is available locally.
3. Update the dataset path in the first notebook cell.
4. Install the required libraries.
5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Iris_flower_classification.ipynb`.
7. Run the cells in order.
8. Inspect the missing-value check, violin plots, pair plot, cluster visualizations, classification report, and confusion matrix.

## Project Structure

A recommended repository layout is:

```text
Iris-Flower-Classification/
│
├── Iris_flower_classification.ipynb
├── README.md
└── requirements.txt
```

- `Iris_flower_classification.ipynb` — data exploration, K-Means clustering, model training, and evaluation.
- `README.md` — project documentation.
- `requirements.txt` — optional file listing tested package versions.

This is a suggested repository layout; the notebook does not generate the README or `requirements.txt` automatically.

## Limitations and Improvements

### 1. Cluster Feature Is Created Before Splitting

The notebook fits K-Means on the full dataset, assigns a cluster to every row, and then splits those rows into training and test sets. The cluster feature is therefore calculated using information from all observations, including the later test observations.

**Improvement:** perform the split first and fit K-Means only on training data, then assign clusters to validation and test data using the fitted clustering model. Alternatively, omit the cluster feature from supervised classification if it does not offer demonstrable value.

### 2. The Evaluation Set Is Small

The saved classification report evaluates only 30 examples. A single small split can produce an optimistic score and may not represent performance across different splits.

**Improvement:** use stratified splitting and repeated stratified cross-validation. Keep a final untouched test set for one final evaluation.

### 3. Cluster IDs Are Not Species Labels

K-Means IDs do not have a fixed semantic meaning. Cluster `0` is not inherently `Iris-setosa`, `Iris-versicolor`, or `Iris-virginica`.

**Improvement:** use cluster visualization and external evaluation measures such as Adjusted Rand Index for clustering, and retain the actual species labels for supervised classification.

### 4. Classification Output Uses Numeric Labels

The saved classification report displays class IDs rather than species names.

**Improvement:** use `classification_report(y_test, y_pred, target_names=label_encoder.classes_)` and provide labels to the confusion-matrix plot.

### 5. K-Means Configuration Is Not Fully Reproducible

The K-Means models omit `random_state` and `n_init`, and the saved output includes a FutureWarning about `n_init`.

**Improvement:** specify `random_state=42` and `n_init=10` explicitly.

### 6. Scaling Step Is Not Used by the Final Classifier

The dataframe is scaled in one cell but later reset from the original dataframe before building the final classification features.

**Improvement:** remove unused scaling cells or clearly separate clustering experiments from the classification pipeline. If a different model requires scaling, fit the scaler on training data only and apply the same transformation at inference time.

### 7. No Model Export or External Prediction Workflow

The notebook trains and evaluates the classifier but does not save the fitted model or demonstrate prediction on an external flower measurement.

**Improvement:** save the trained model with `joblib` and implement a small inference function that accepts the four measurements in the exact feature order expected by the model.

## Conclusion

This project demonstrates exploratory analysis, unsupervised clustering, and supervised classification on the Iris flower dataset. It visualizes the distributions and relationships of sepal and petal measurements, uses K-Means to assign three clusters, and applies a Random Forest classifier to predict species.

The saved notebook reports **100.00% accuracy on a 30-example test set**, with a precision, recall, and F1-score of 1.00 for each encoded class. However, the cluster feature is generated from the full dataset before splitting, and the test set is small. A more reliable evaluation should create the train/test split first, fit feature engineering only on training data, use stratified evaluation, and repeat the experiment with cross-validation.

---

**Reproducibility note:** The metrics in this README reflect the notebook's saved output and may differ when rerunning it with a different scikit-learn version, clustering initialization, or data split.
