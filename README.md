# Machine Learning Projects

A collection of machine learning projects demonstrating data preprocessing, exploratory data analysis, classification, clustering, and model evaluation using Python and Scikit-learn.

## Projects Overview

| Project | Algorithm | Recorded Accuracy |
|---|---|---:|
| Credit Card Fraud Detection | Logistic Regression with SMOTE | 97.48% |
| Titanic Survival Prediction | Decision Tree Classifier | 78.21% |
| Iris Flower Classification | Random Forest and K-Means | 100.00% |

*Note: These results come from the saved notebook outputs. Evaluation limitations mean the scores should not be directly compared or interpreted as guaranteed performance on new data.*

## 1. Credit Card Fraud Detection

**Objective:** Identify potentially fraudulent credit card transactions.

- **Dataset:** 284,807 transactions, including 492 fraud cases.
- **Techniques:** Data preprocessing, feature scaling, SMOTE oversampling, and Logistic Regression.
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix.
- **Recorded accuracy:** 97.48% on a resampled evaluation set.

**Improvement needed:** Apply SMOTE only after splitting the data to prevent data leakage and evaluate on an untouched test set with the original class distribution.

## 2. Titanic Survival Prediction

**Objective:** Predict passenger survival based on passenger class, age, sex, family information, fare, and embarkation port.

- **Dataset:** 891 passenger records.
- **Algorithm:** Decision Tree Classifier.
- **Preprocessing:** Missing-value imputation and categorical encoding.
- **Evaluation:** Accuracy, precision, recall, and F1-score.
- **Recorded accuracy:** 78.21% on 179 test passengers.

**Improvement needed:** Fit preprocessing on training data only and use a stratified train/test split for more reliable evaluation.

## 3. Iris Flower Classification

**Objective:** Classify Iris flower species using sepal and petal measurements.

- **Features:** Sepal length, sepal width, petal length, and petal width.
- **Algorithms:** K-Means clustering and Random Forest classification.
- **Evaluation:** Accuracy, classification report, and confusion matrix.
- **Recorded accuracy:** 100% on 30 test samples.

**Improvement needed:** Fit K-Means using training data only before generating cluster features, and validate performance using cross-validation or a larger independent test set.

## Technologies Used

- **Programming:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Data balancing:** imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **Development environment:** Jupyter Notebook

## General Workflow

1. Load and explore the dataset.
2. Clean and preprocess data.
3. Engineer and transform features.
4. Train machine learning models.
5. Evaluate performance using appropriate metrics.
6. Identify limitations and potential improvements.

## Installation

Install the primary dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter imbalanced-learn
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the relevant project notebook and run its cells in order. Update local dataset paths where required.

## Future Improvements

- Prevent data leakage during preprocessing and resampling.
- Use stratified splits and cross-validation where appropriate.
- Compare multiple algorithms using consistent evaluation methods.
- Add detailed confusion matrices and class-wise metrics.
- Save trained models and preprocessing pipelines.
- Record reproducible dependency versions and inference benchmarks.

## Conclusion

These three projects demonstrate practical applications of machine learning in financial fraud detection, passenger survival prediction, and flower species classification. They highlight the importance of data preprocessing, model selection, and careful performance evaluation.

The recorded accuracy values are preliminary results from the supplied notebooks. Correcting evaluation limitations is essential before making reliable generalization claims.
