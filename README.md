# Machine Learning Model Evaluation and Hyperparameter Tuning
[**`Kaggle Notebook Link `**](https://www.kaggle.com/code/recepbattal/financial-approval-evaluations/notebook)

### Overview

This repository contains a set of scripts and code snippets for evaluating and tuning machine learning models on a dataset. The process includes data preprocessing, model training, hyperparameter tuning, and performance evaluation.


## Contents

1-Data Preparation

2-Exploratory Data Analysis (EDA)

3-Categorical and Numerical Data Processing

4-Feature Scaling and Balancing

5-Model Training and Evaluation

6-Hyperparameter Tuning with Grid Search

7-Model Performance Evaluation


### 1. Data Preparation
This section involves loading and preparing the data for analysis.
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
arecords = pd.read_csv('application_records.csv')
crecords = pd.read_csv('credit_records.csv')

# Create 'accountAge' from 'MONTHS_BALANCE'
crecords['accountAge'] = crecords['MONTHS_BALANCE'] * -1

# Display columns and info
print(arecords.columns)
print(crecords.columns)
print(arecords.info())
print(crecords.info())
```

### 2. Exploratory Data Analysis (EDA)
Perform initial exploratory data analysis to understand the dataset's structure and content.
```python
# Inspecting data
print(arecords.head(10))
print(crecords.head(10))

# Check for missing values
print(arecords.isna().sum())
print(crecords.isna().sum())

# Visualize missing values
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(arecords.isna(), color='red')
sns.heatmap(crecords.isna(), color='red')
plt.show()
```

### 3. Categorical and Numerical Data Processing
Convert categorical variables to numerical format and handle any data cleaning tasks.
```python
from sklearn.preprocessing import LabelEncoder

# List categorical and numerical columns
categorical_columns = arecords.select_dtypes(include='object').columns.tolist()
numerical_columns = arecords.select_dtypes(exclude='object').columns.tolist()

# Convert categorical columns to numerical
categorical_columns = ['gender', 'car', 'reality', 'incmTp', 'eduTp', 'familyTp', 'houseTp']
label_encoders = {}

for column in categorical_columns:
    if column in arecords.columns:
        le = LabelEncoder()
        arecords[column] = le.fit_transform(arecords[column].astype(str))
        label_encoders[column] = le

print(arecords[categorical_columns].head())
print(arecords.describe(include='all'))
```

### 4. Feature Scaling and Balancing
Scale features and balance the dataset using SMOTE.
```python
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split data
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
mms = MinMaxScaler()
X_scl = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)
X_test_scl = pd.DataFrame(mms.transform(X_test), columns=X_test.columns)

# Apply SMOTE
oversample = SMOTE()
X_blc, y_blc = oversample.fit_resample(X_scl, y_train)
X_test_blc, y_test_blc = oversample.fit_resample(X_test_scl, y_test)
```

### 5. Model Training and Evaluation
Train and evaluate various classifiers on the balanced dataset.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

classifiers = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LightGBM": LGBMClassifier(),
    "LogisticRegression": LogisticRegression(),
}

train_results = []
test_results = []

for key, classifier in classifiers.items():
    classifier.fit(X_blc, y_blc)
    train_result = classifier.score(X_blc, y_blc)
    train_results.append(train_result)
    test_result = classifier.score(X_test_blc, y_test_blc)
    test_results.append(test_result)

print("Train Results:\n", train_results)
print("Test Results:\n", test_results)
```

### 6. Hyperparameter Tuning with Grid Search
Optimize hyperparameters for the CatBoost model using Grid Search.
```python
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

param_grid = {
    'iterations': [10, 25, 50],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5]
}

catboost_model = CatBoostClassifier(verbose=0)
grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_blc, y_blc)

print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)
```

### 7. Model Performance Evaluation
Evaluate the performance of the best model using cross-validation and generate classification reports.
```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X_blc, y_blc, cv=5, scoring='accuracy')
print("Cross-validation results:", cv_scores)
print("Average CV accuracy:", cv_scores.mean())

# Classification report
best_catboost_model = grid_search.best_estimator_
preds = best_catboost_model.predict(X_test_blc)
print(classification_report(y_test_blc, preds))
```

## Installation
To install the required libraries, you can use:
```bash
pip install catboost lightgbm imbalanced-learn
```
