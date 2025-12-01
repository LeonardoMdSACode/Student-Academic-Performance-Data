## About AcademicPerformance-FinalMarks.ipynb

# 🎓 Student Academic Performance Predictor

This project analyzes a student academic performance dataset and employs multiple Machine Learning regression algorithms to predict a student's final marks based on their demographic, academic, and social factors.

---

## 🚀 Project Overview

The core objective is to:

1. **Visualize** the relationships between different factors (columns) in the dataset.
2. **Clean** and preprocess the data for model readiness.
3. **Train and Compare** eight different regression models to determine the most accurate predictor of student final marks.

The final output is a visualization of the $\text{R}^2$ scores and Mean Squared Errors (MSE) for all models, allowing for an immediate assessment of the best-performing algorithm.

---

## 🛠️ Data Preprocessing and Analysis

### 1. Data Loading and Cleaning

The initial steps focus on preparing the dataset:

* **Load Data:** Read the `Final_Marks_Data.csv`.
* **Drop Identifier:** The non-predictive `Student_ID` column is dropped (`df.drop("Student_ID", axis=1, inplace=True)`).
* **Handle Missing Values:** Any row containing missing data (`NaN`) is removed using `df.dropna(inplace=True)`.

### 2. Exploratory Data Analysis (EDA)

* **Relationship Visualization:** A **pair plot** is generated using `sns.pairplot(df, vars=df.columns)` to visually inspect the relationships, distributions, and potential correlations between every pair of columns in the cleaned dataset. This step helps in understanding which features might be strong predictors of the final mark.

---

## 🧠 Machine Learning Model Training

### 1. Feature and Target Separation

The dataset is divided into features ($X$) and the target variable ($Y$) for training:

* $X$: Features (`df.iloc[:, :-1].values`) - All columns **except the last one**. These are the input factors (e.g., prior grades, attendance).
* $Y$: Target (`df.iloc[:, -1].values`) - The **last column** (the final marks/grade).

### 2. Train-Test Split

The data is split to ensure the model's performance is evaluated on unseen data:

* **Training Set (80%)**: Used to teach the models the underlying patterns.
* **Test Set (20%)**: Reserved for evaluating the model's accuracy after training.
* A `random_state=42` is used for reproducibility.

### 3. Regression Models

The `training_regression()` function compares the performance of the following eight models:

| Category | Model Name | Abbreviation |
| :--- | :--- | :--- |
| **Ensemble** | Random Forest Regressor | `rfr` |
| | AdaBoost Regressor | `abr` |
| | Gradient Boosting Regressor | `gbr` |
| | Extra Trees Regressor | `etr` |
| **Boosted** | XGBoost Regressor | `xgbr` |
| | LightGBM Regressor | `lgbr` |
| **Linear** | Linear Regression | `lnr` |
| **Kernel-based**| Support Vector Regressor | `svr` |

---

## 📊 Model Evaluation and Results

The `training_regression()` function performs the following steps:

1. **Fit** each model to the training data ($\text{x\_train}$, $\text{y\_train}$).
2. **Predict** on the test data ($\text{x\_test}$).
3. Calculate the two main performance metrics:
    * **$\text{R}^2$ Score:** A measure of how well the regression predictions approximate the real data points. Higher is better (closer to 100%).
    * **Mean Squared Error (MSE):** The average squared difference between the estimated values and the actual value. Lower is better (closer to 0).
4. Generate two bar plots comparing the $\text{R}^2$ scores and MSEs across all eight models. The best model will have the highest $\text{R}^2$ and the lowest MSE.
The results are sorted by $\text{R}^2$ score, showing the top-performing models first.

### Performance Metrics Summary

| Model Name | $\text{R}^2$ Score ($\%$) | MSE | Rank (by $\text{R}^2$) |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | **79.49 %** | 21.3254 | **1** |
| Gradient Boosting | 78.04 % | 22.9101 | 2 |
| LightGBM | 75.34 % | 25.9967 | 3 |
| Random Forest | 73.31 % | 27.2053 | 4 |
| XGBoost | 72.55 % | 29.9313 | 5 |
| Extra Trees | 71.86 % | 29.5187 | 6 |
| Ada Boost | 67.62 % | 28.5175 | 7 |
| Support Vector Machine | 60.24 % | 28.4099 | 8 |

### Key Findings

* The **Linear Regression** model demonstrated the best performance, explaining **79.49%** of the variance in the student's final marks, and achieving the lowest Mean Squared Error (MSE) of **21.33**.
* The **Gradient Boosting** and **LightGBM** ensemble methods also performed strongly, indicating that the relationship between the features and the target variable is generally well-suited for linear or highly regularized tree-based models.

---

## 🐍 Prerequisites

This project requires the following Python libraries:

* `numpy`
* `pandas`
* `warnings`
* `datetime`
* `matplotlib`
* `seaborn`
* `scikit-learn` (sklearn)
* `xgboost`
* `lightgbm`

You can install all necessary packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

---
## About AcademicPerformance-Tensorflow.ipynb

# 🎓 Student Performance Prediction using Deep Learning

This project implements a **Deep Neural Network (DNN)** using **TensorFlow/Keras** to predict student final exam marks based on various academic and study-related features.


---

## 🎯 Project Goals

1.  **Data Preparation:** Load, clean, and standardize the synthetic student performance dataset.
2.  **Model Development:** Construct a **Feed-Forward Neural Network** for regression.
3.  **Training:** Train the model using the Adam optimizer and Mean Squared Error (MSE) loss.
4.  **Evaluation:** Assess the model's accuracy using the Mean Absolute Error (MAE) on a dedicated test set.
5.  **Prediction Demonstration:** Apply the trained model to predict the score for a new, hypothetical student.

---

## 💻 Dependencies and Setup

This script requires the following standard Python libraries. It is highly recommended to use a virtual environment.

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib

## 📊 Data Features and Preprocessing

The model uses **5 input features** to predict the **Final Exam Marks (out of 100)**:

| Feature ($\mathbf{X}$) | Target ($\mathbf{y}$) |
| :--- | :--- |
| Attendance (%) | Final Exam Marks (out of 100) |
| Internal Test 1 (out of 40) | |
| Internal Test 2 (out of 40) | |
| Assignment Score (out of 10) | |
| Daily Study Hours | |

### Preprocessing Steps

* **Train-Test Split:** Data is divided into **80% for training** and **20% for testing** to ensure the model's performance is measured on unseen data.
* **Standardization:** Features are scaled using `StandardScaler`. This transforms the data such that it has a **mean of 0** and a **standard deviation of 1** ($$\frac{X - \mu}{\sigma}$$). This step is vital for neural networks as it aids in faster convergence and prevents issues like vanishing/exploding gradients.

---

## 🧠 Model Architecture & Training

The model is a **Sequential Deep Neural Network**:

### Architecture

| Layer Type | Neurons | Activation | Purpose |
| :--- | :--- | :--- | :--- |
| **Dense (Hidden 1)** | 64 | **ReLU** | Primary feature learning layer. |
| **Dense (Hidden 2)** | 32 | **ReLU** | Captures more abstract patterns. |
| **Dropout** | (N/A) | 20% rate | **Regularization** to prevent overfitting by randomly dropping nodes during training. |
| **Dense (Output)** | 1 | **Linear** | Outputs the continuous predicted score. |

### Compilation and Training Parameters

* **Optimizer:** `adam`
* **Loss Function (for Training):** **Mean Squared Error (MSE)**, minimizing the square of the difference between actual and predicted marks.
    * $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Training Epochs:** 100
* **Batch Size:** 16

---

## 📈 Evaluation and Results

The model was evaluated on the **20% held-out test set**.

### Performance Metrics

| Metric | Formula | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE** | $$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$ | **$4.06$ Marks** | On average, the prediction is off by only **$\pm 4.06$ marks**. |
| **Loss (MSE)** | $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ | $24.73$ | The primary loss minimized during training. |

Test Set Mean Squared Error (MSE): 24.67
Test Set R^2 Score: 80.30% (highest value so far, higher than linear regression!)

### Sample Predictions

| Actual Marks ($\mathbf{y}$) | Predicted Marks ($\mathbf{\hat{y}}$) | Difference |
| :---: | :---: | :---: |
| 62 | 61.64 | 0.36 |
| 78 | 79.52 | -1.52 |
| 59 | 59.85 | -0.85 |
| 69 | 67.24 | 1.76 |
| 66 | 66.86 | -0.86 |

---

## ✅ Live Prediction Demonstration

A new student with the following excellent metrics was submitted for prediction:

* **Inputs:** \[85% Attendance, 30/40 Test 1, 32/40 Test 2, 8/10 Assignment, 3 Study Hours]
* **Predicted Final Exam Score:** **$64.91$**
