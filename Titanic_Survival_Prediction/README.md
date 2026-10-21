# Titanic Survival Prediction Using Decision Tree

A machine learning project that predicts whether a passenger survived the Titanic disaster using passenger attributes such as passenger class, sex, age, family relationships, ticket fare, and port of embarkation. The notebook preprocesses the Titanic dataset, encodes categorical fields, trains a Decision Tree classifier, and evaluates its predictions using accuracy and a classification report.

> **Recorded notebook result:** 78.21% accuracy on 179 test passengers. The notebook's metrics are documented as recorded. Some preprocessing steps are performed before the train/test split, so the evaluation can be improved by fitting preprocessing only on the training data and using a stratified split.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Model: Decision Tree Classifier](#model-decision-tree-classifier)
- [Accuracy and Test Results](#accuracy-and-test-results)
- [Efficiency and Performance](#efficiency-and-performance)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Conclusion](#conclusion)

## Project Overview

The Titanic Survival Prediction project is a supervised binary-classification task. It uses passenger information to predict the `Survived` label:

- `0` — did not survive
- `1` — survived

The notebook reads a Titanic CSV dataset, handles missing values, removes selected columns, converts categorical variables into numeric values, splits the data into training and test sets, and trains a Decision Tree classifier.

This is an educational machine learning project. Its predictions describe patterns learned from the historical dataset; they should not be interpreted as causal explanations of survival.

## Objectives

- Load and examine the Titanic passenger dataset.
- Identify and handle missing values.
- Select relevant input features and remove columns not used by the model.
- Convert categorical variables into numeric values.
- Train a Decision Tree classifier.
- Evaluate predictions using accuracy, precision, recall, F1-score, and support.
- Identify practical improvements that could make the evaluation more reliable.

## Dataset

The notebook loads a CSV file from a local ZIP archive using Pandas. The path currently in the notebook is an absolute Windows path and must be updated to match your own computer.

The dataset contains **891 passenger records** with 12 original columns.

| Column | Description |
|---|---|
| `PassengerId` | Passenger record identifier |
| `Survived` | Target: 0 = did not survive, 1 = survived |
| `Pclass` | Passenger ticket class |
| `Name` | Passenger name |
| `Sex` | Passenger sex as supplied in the dataset |
| `Age` | Passenger age |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Ticket` | Ticket identifier |
| `Fare` | Passenger fare |
| `Cabin` | Cabin identifier, where recorded |
| `Embarked` | Port of embarkation |

After preprocessing, the notebook retains eight columns: the target `Survived` plus seven input features.

## Features Used

The notebook uses these seven features to predict survival:

| Feature | Meaning |
|---|---|
| `Pclass` | Passenger class, which provides information about ticket class |
| `Sex` | Encoded passenger sex |
| `Age` | Passenger age, with missing values imputed |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Fare` | Ticket fare |
| `Embarked` | Encoded embarkation port |

The model does not use `PassengerId`, `Name`, `Ticket`, or `Cabin`; those columns are dropped during preprocessing.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Loading and manipulating the dataset |
| Scikit-learn | Imputation, label encoding, train/test split, model training, and metrics |
| Matplotlib | Plotting |
| Seaborn | Visualizing survival counts |
| Jupyter Notebook | Interactive experimentation |

## Project Workflow

1. **Load data:** read the dataset from the ZIP archive into a Pandas dataframe.
2. **Inspect records:** preview passenger data and count missing values.
3. **Impute age:** fill missing `Age` values using the median.
4. **Impute embarkation port:** fill missing `Embarked` values using the most frequent category.
5. **Remove columns:** drop `PassengerId`, `Name`, `Ticket`, and `Cabin`.
6. **Encode categories:** convert `Sex` and `Embarked` into numeric labels.
7. **Separate target and features:** assign `Survived` as the target and the remaining columns as the inputs.
8. **Split the dataset:** reserve 20% of rows for testing.
9. **Train the classifier:** fit a Decision Tree using the training data.
10. **Evaluate predictions:** calculate accuracy and print the classification report.
11. **Visualize target counts:** plot the count of passengers in each survival class.

## Data Preprocessing

### 1. Missing-Value Analysis

The original notebook reports these missing values:

| Column | Missing values |
|---|---:|
| `Age` | 177 |
| `Cabin` | 687 |
| `Embarked` | 2 |
| Other columns | 0 |

Missing values need to be addressed before the selected model is trained.

### 2. Age Imputation

The notebook uses `SimpleImputer(strategy="median")` to fill missing ages with the median age:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
titanic["Age"] = imputer.fit_transform(titanic[["Age"]])
```

The median is less sensitive to unusually high or low values than the mean. In the notebook, imputation is performed before the train/test split. For a stricter evaluation, the imputer should be fitted on the training partition only, then used to transform the test partition.

### 3. Embarked Imputation

Two rows have missing `Embarked` values. The notebook fills these values using the most frequent category, shown in the output as `S`:

```python
most_frequent_embarked = titanic["Embarked"].mode()[0]
titanic["Embarked"].fillna(most_frequent_embarked, inplace=True)
```

After this operation, the saved output reports no remaining missing values in `Embarked`.

### 4. Column Removal

The notebook drops these columns:

```python
titanic.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"],
    inplace=True
)
```

`Cabin` has many missing values. The identifier and text columns are also removed, leaving the target and seven input features.

Removing these columns simplifies the baseline model, although `Name`, `Ticket`, or cabin-related information could be explored with more advanced feature engineering in a future version.

### 5. Categorical Encoding

The notebook applies `LabelEncoder` to `Sex` and `Embarked`.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
titanic["Sex"] = label_encoder.fit_transform(titanic["Sex"])
titanic["Embarked"] = label_encoder.fit_transform(titanic["Embarked"])
```

The saved sample shows `female` encoded as `0`, `male` as `1`, and the embarkation labels encoded as integers.

For this tree-based model, numeric encoding is workable, but `LabelEncoder` imposes arbitrary integer codes on nominal categories. For a more maintainable preprocessing workflow, use a fitted `OrdinalEncoder` or `OneHotEncoder` in a Scikit-learn pipeline, chosen according to the model and feature meaning.

## Model: Decision Tree Classifier

The notebook uses Scikit-learn's `DecisionTreeClassifier`:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
```

A Decision Tree learns a sequence of feature-based rules. At each split, it divides examples according to a feature and threshold or category code, aiming to separate the target classes. Predictions are made by following the learned rules from the root to a leaf.

### Why a Decision Tree?

- It can learn nonlinear relationships and interactions between features.
- It can handle numerical input features without standardization.
- Its decision rules can be inspected and visualized.
- It is a useful baseline for small tabular datasets.

### Train/Test Split

The notebook uses an 80/20 split with `random_state=42`:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

The test set contains **179 passengers**, as shown by the classification report. The split does not set `stratify=y`, so class proportions are not explicitly preserved across the training and test partitions.

## Accuracy and Test Results

The saved notebook prints an accuracy of **78.21%** on the held-out test set.

| Metric | Class 0: Did not survive | Class 1: Survived |
|---|---:|---:|
| Precision | 0.83 | 0.72 |
| Recall | 0.79 | 0.77 |
| F1-score | 0.81 | 0.75 |
| Support | 105 | 74 |

Overall metrics from the notebook:

| Metric | Recorded result |
|---|---:|
| Test accuracy | **78.21%** |
| Test samples | **179** |
| Macro-average precision | 0.78 |
| Macro-average recall | 0.78 |
| Macro-average F1-score | 0.78 |
| Weighted-average precision | 0.79 |
| Weighted-average recall | 0.78 |
| Weighted-average F1-score | 0.78 |

### Accuracy

Accuracy is the fraction of predictions that match the actual labels. The recorded 78.21% result means the model classifies roughly 78 of every 100 passengers correctly in this test split.

Accuracy alone does not describe all errors. Precision indicates how often predictions for a class are correct, while recall indicates how many examples of that class are found. F1-score balances precision and recall.

### Interpreting the Class-Wise Results

For class `0` (did not survive), the notebook reports precision of 0.83 and recall of 0.79. For class `1` (survived), it reports precision of 0.72 and recall of 0.77.

These metrics indicate that the model's predictions differ in quality between the two classes. The model's performance should be evaluated according to the intended use and error costs rather than accuracy alone.

The notebook does not save a confusion-matrix output or detailed feature-importance analysis. These would be useful additions for further model interpretation.

## Efficiency and Performance

### Computational Characteristics

A Decision Tree is usually inexpensive to train and predict with on a dataset of this size. It does not require feature standardization, and inference consists of following a path through the trained tree.

However, the notebook does **not** record model training time, inference latency, memory usage, tree depth, node count, or saved model size. Therefore, precise timing or deployment-efficiency claims cannot be made from the saved outputs.

### Practical Improvements

- Measure model fit time and prediction time using `time.perf_counter()`.
- Inspect and limit `max_depth`, `min_samples_leaf`, or `min_samples_split` to control complexity and potentially reduce overfitting.
- Compare the tree with Logistic Regression, Random Forest, and Gradient Boosting using the same data split.
- Use cross-validation to estimate performance across multiple partitions.
- Add a confusion matrix and feature-importance analysis.
- Save the trained model and preprocessing components together so future predictions use the same transformations.

## Installation

### Prerequisites

- Python 3.10 or another compatible Python version.
- Jupyter Notebook or JupyterLab.
- The Titanic passenger dataset file used by the notebook.
- The Python packages listed below.

Install the dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

### Configure the Dataset Path

The notebook currently reads a dataset from an absolute Windows path:

```python
titanic = pd.read_csv(
    r"C:\Users\rahul\OneDrive\Desktop\Codsoft\Titanic_Survival_Prediction\archive.zip"
)
```

Update this path to match your environment. For example, if the dataset archive is placed in a local `data` directory:

```python
titanic = pd.read_csv("data/archive.zip")
```

The relative path is an example; ensure that the archive contains the expected CSV and use the correct path for your downloaded file.

## How to Run

1. Clone or download the repository.
2. Download the Titanic dataset from its authorized source.
3. Place the data archive in the expected location and update the path in the notebook.
4. Install the required packages.
5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Titanic_Model.ipynb`.
7. Execute the cells in order.
8. Review the missing-value counts, preprocessing output, target-count plot, accuracy, and classification report.

## Project Structure

A suggested repository structure is:

```text
Titanic-Survival-Prediction/
│
├── Titanic_Model.ipynb
├── README.md
├── requirements.txt
└── data/
    └── archive.zip   # Local dataset; do not commit unless permitted
```

- `Titanic_Model.ipynb` — preprocessing, model training, prediction, and evaluation.
- `README.md` — project documentation.
- `requirements.txt` — optional list of tested dependency versions.
- `data/` — suggested location for a local dataset file.

The `data/` directory and `requirements.txt` are suggestions and are not generated by the notebook itself.

## Limitations and Future Improvements

### 1. Fit Preprocessing Only on Training Data

The notebook imputes missing values and encodes categorical fields before the train/test split. In particular, the median age and most-frequent embarkation category are calculated from the full dataset.

**Improvement:** split the raw dataset first, fit imputers and encoders on the training data, and apply the fitted transformations to the test data through a Scikit-learn pipeline.

### 2. Use a Stratified Split

The `train_test_split` call does not use `stratify=y`. The test subset may therefore have class proportions that differ from the full dataset.

**Improvement:** use `train_test_split(..., stratify=y, random_state=42)` to preserve the target proportions more consistently.

### 3. Tune and Validate the Decision Tree

The notebook uses default Decision Tree parameters apart from `random_state=42`. A deep tree can fit idiosyncrasies in the training data and generalize less well.

**Improvement:** evaluate tree depth and minimum-sample settings with cross-validation, and compare the results against simple baselines and ensemble models.

### 4. Improve Model Interpretation

The notebook reports accuracy and a classification report but does not calculate the confusion matrix or feature importance.

**Improvement:** plot a confusion matrix and inspect feature importances, while remembering that feature importance is not proof of a causal relationship.

### 5. Save the Full Inference Workflow

The notebook does not demonstrate exporting a trained model along with its preprocessing steps.

**Improvement:** use a pipeline for imputers, encoders, and the classifier, then persist the complete fitted pipeline with `joblib`.

### 6. Recognize Dataset Limitations

The Titanic dataset is a historical dataset. It has a limited number of records and reflects circumstances specific to that event. Its learned patterns should not be treated as universal rules about survival or human behavior.

**Improvement:** explain that the project is an educational prediction exercise and report the limitations of the sample, features, and evaluation.

## Conclusion

This project demonstrates an end-to-end baseline classification workflow using the Titanic passenger dataset and a Scikit-learn Decision Tree. It handles missing values, removes unused columns, encodes categorical fields, trains a binary classifier, and evaluates it on a held-out subset.

The notebook records **78.21% accuracy** on **179 test passengers**. The classification report gives class-wise precision, recall, F1-score, and support, offering more context than accuracy alone.

The most useful next steps are to fit preprocessing only on training data, use a stratified split, validate model complexity, add a confusion matrix and feature-importance analysis, and compare the Decision Tree with alternative classifiers.

---

**Reproducibility note:** The reported metrics describe the saved notebook run. Results may differ with changes to the data, split strategy, preprocessing, library versions, or model hyperparameters.
