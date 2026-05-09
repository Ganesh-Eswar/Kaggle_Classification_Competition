# Hotel Booking Cancellation Prediction using Machine Learning

## Overview

Explanation video: https://drive.google.com/file/d/1ZLn1HPZDyidyoeEtHT4P-IksnhDZgpef/view?usp=sharing

This project focuses on predicting whether a hotel customer will cancel their booking based on customer behavior, booking details, room preferences, pricing, and reservation characteristics.

The project was developed as part of a Kaggle machine learning classification competition and includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Missing value handling
* Feature encoding
* Outlier removal
* Feature scaling
* Multiple classification models
* Ensemble learning techniques
* Hyperparameter tuning

The primary objective is to accurately classify bookings into:

* `1` → Cancelled
* `0` → Not Cancelled

---

# Dataset Description

The dataset contains hotel booking information and customer reservation details.

## Files

* `train.csv` → Training dataset containing features and target variable
* `test.csv` → Test dataset with hidden target values
* `sample_submission.csv` → Sample Kaggle submission format

---

# Features Used

| Feature          | Description                                 |
| ---------------- | ------------------------------------------- |
| `id`             | Unique identifier                           |
| `adults`         | Number of adults                            |
| `children`       | Number of children                          |
| `weekends`       | Weekend stay duration                       |
| `weekdays`       | Weekday stay duration                       |
| `meal_type`      | Selected meal plan                          |
| `room_type`      | Room category selected                      |
| `arrival`        | Customer arrival date                       |
| `lead_time`      | Days between booking and arrival            |
| `segment`        | Booking segment/category                    |
| `repeat`         | Whether customer previously booked          |
| `price`          | Average room price                          |
| `requests`       | Number of special requests                  |
| `booking_status` | Target variable (Cancelled / Not Cancelled) |

---

# Project Workflow

# 1. Data Loading

Datasets were loaded using:

* Pandas
* Kaggle notebook environment

---

# 2. Exploratory Data Analysis (EDA)

EDA techniques included:

* Column-wise bar chart analysis
* Scatter plots
* Distribution understanding
* Missing value analysis
* Categorical feature inspection

Visualization libraries:

* Matplotlib
* Seaborn

---

# 3. Data Preprocessing

## Missing Value Handling

Missing values were identified and handled using:

* `SimpleImputer(strategy="most_frequent")`
* `SimpleImputer(strategy="median")`

Columns handled:

* `room_type`
* `lead_time`
* `price`

---

## Feature Selection

The `arrival` column was removed because:

* `lead_time` already captures booking timing information effectively.

---

# 4. Categorical Encoding

Used:

* `OneHotEncoder`
* `ColumnTransformer`

Encoded features:

* `meal_type`
* `room_type`
* `segment`

Unknown categories were handled using:

```python
handle_unknown='ignore'
```

---

# 5. Duplicate Removal

* Dropped unnecessary `id` column
* Removed duplicate rows from training data

---

# 6. Outlier Detection and Removal

Implemented custom IQR-based outlier removal for numerical columns.

Outliers were removed using:

* Interquartile Range (IQR) method
* Statistical boundary filtering

---

# 7. Feature Scaling

Applied:

* `StandardScaler`

to normalize numerical feature distributions before training.

---

# Machine Learning Models Used

Multiple classification algorithms were implemented and compared.

---

# Base Models

## Tree-Based Models

* Random Forest Classifier
* Decision Tree Classifier
* Gradient Boosting Classifier
* Extra Trees Classifier

---

## Boosting Models

* XGBoost Classifier
* CatBoost Classifier

---

## Linear Models

* Logistic Regression

---

## Distance / Kernel-Based Models

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

---

# Ensemble Learning

## Voting Classifier

Combined:

* Logistic Regression
* SVM
* KNN
* Random Forest
* Gradient Boosting

using:

```python
voting='soft'
```

---

## Stacking Classifier

Implemented advanced stacking ensemble using:

* Random Forest
* Gradient Boosting
* Extra Trees
* SVM
* KNN

Final estimator:

* Logistic Regression

---

# Hyperparameter Tuning

Used:

* `GridSearchCV`
* Cross-validation

for tuning:

* Random Forest
* Logistic Regression
* Decision Tree

---

# Evaluation Metric

Model performance was evaluated primarily using:

* Accuracy Score

Additional imported metrics:

* Classification Report

---

# Submission Files Generated

Several prediction submission files were generated, including:

```text
catboost_out_rm.csv
xgboost_prop_rmo.csv
random_forest_rmo.csv
votting_classifier_rmo.csv
stack_classifier_rmo.csv
hyp_rf.csv
hyp_logistic.csv
hyp_decisiontree.csv
```

---

# Technologies Used

## Programming Language

* Python

## Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* CatBoost

---

# Project Structure

```text
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
├── outputs/
│   ├── catboost_classifier.csv
│   ├── xgboost_classifier.csv
│   ├── voting_classifier.csv
│   └── ...
├── README.md
├── Demonstration_video.mp4
└── requirements.txt
```

---

# Key Learning Outcomes

Through this project:

* Binary classification workflows were explored
* Ensemble learning techniques were implemented
* Stacking and voting classifiers were developed
* Hyperparameter optimization was practiced
* Real-world customer behavior prediction was analyzed
* End-to-end ML pipeline development was performed

---

# Future Improvements

Potential future enhancements include:

* ROC-AUC optimization
* Feature importance analysis
* SHAP explainability
* SMOTE for imbalance handling
* LightGBM integration
* Automated ML pipelines
* Cross-validation based leaderboard optimization

---

# Author

Developed as part of a Kaggle Machine Learning Competition focused on hotel booking cancellation prediction using classification and ensemble learning techniques.
