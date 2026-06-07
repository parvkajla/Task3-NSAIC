# Task 3: Machine Learning - The "Baseline Beater"

## The Scenario

An intern left behind a machine learning script (`baseline_model.ipynb`) that technically runs, but its performance was mediocre. The goal was to improve the model's ability to predict customer responses to a marketing campaign and explain the reasoning behind the improvements.

## The Objective

The baseline script predicts customer behavior using the `Response` column as the target variable. The objective was to improve the F1-Score by at least **20%** relative to the baseline.

---

## Baseline Performance

| Metric   | Score |
| -------- | ----- |
| F1-Score | 0.282 |

The baseline implementation:

* Used Logistic Regression
* Dropped all categorical features
* Replaced missing values with 0
* Did not explicitly address class imbalance

---

## Improvements Implemented

### 1. Preserved Categorical Features

The baseline removed all object-type columns such as:

* `Education`
* `Marital_Status`
* `Dt_Customer`

These features were retained using **One-Hot Encoding** so that useful customer information could contribute to model predictions.

### 2. Improved Missing Value Handling

The dataset contained missing values in the `Income` column.

Instead of replacing missing values with 0, I used:

* Median Imputation for numerical features
* Most Frequent Imputation for categorical features

This preserves the underlying data distribution more effectively.

### 3. Handled Class Imbalance

The dataset is highly imbalanced:

| Response | Count |
| -------- | ----- |
| 0        | 1906  |
| 1        | 334   |

To improve prediction of the minority class, I used:

```python
class_weight='balanced'
```

within the model.

### 4. Replaced Logistic Regression with Random Forest

The baseline model was replaced with a tuned Random Forest Classifier:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
```

---

## Write-Up (Mandatory)

### 1. Most Impactful Change

The most impactful change was preserving categorical features through One-Hot Encoding and training a class-balanced Random Forest instead of dropping all non-numeric columns.

### 2. Why This Improved the Model

The baseline Logistic Regression model learned a single linear decision boundary and ignored useful categorical information. Additionally, the dataset was highly imbalanced, causing the model to favor the majority class.

Random Forest combines multiple decision trees and can learn nonlinear relationships and feature interactions. Using class balancing increased the importance of minority-class samples, improving recall and overall F1-Score.

### 3. Final Metric Achieved

| Metric            | Score |
| ----------------- | ----- |
| Baseline F1-Score | 0.282 |
| Final F1-Score    | 0.548 |
| Improvement       | 94.3% |

---

## Evaluation Criteria

This solution focuses not only on improving model performance but also on understanding the reasoning behind each modification. The improvements were chosen based on:

* Data characteristics
* Class imbalance
* Feature importance
* Model suitability for nonlinear relationships

---

## Repository Contents

```text
Task3-NSAIC/
│
├── baseline_model.ipynb
├── marketing_campaign.csv
└── README.md
```

---

## Author

**Parv Kajla**
B.Tech CSE (AI & ML)
Galgotias University
