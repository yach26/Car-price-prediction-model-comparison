# Car Price Prediction: Model Comparison & Analysis

This repository contains a comprehensive machine learning workflow for predicting car prices using a variety of regression models. We evaluate, compare, and analyze the performance of **Linear Regression**, **Decision Trees**, and **Random Forest Regressors** (both baseline and optimized variants).

---

## 📌 Project Overview
Car valuation is a complex regression task influenced by brand reputation, engine specifications, mileage, fuel type, transmission, and body design. The goal of this project is to clean, preprocess, encode, and model a vehicle dataset to predict the price of a car while exploring the bias-variance tradeoff across different algorithms.

---

## 🛠️ Data Preprocessing & Feature Engineering
Before feeding the data to the machine learning models, a rigorous preprocessing pipeline was executed:

1. **Handling Missing Values:**
   - **Target Variable (`price`):** Rows with missing prices (23 instances) were dropped since imputing the target variable introduces artificial bias.
   - **Unstructured Data:** The `description` column was dropped due to high cardinality and text complexity.
   - **Numerical Imputation:** Missing values in `mileage` were filled with the **mean** value of the column.
   - **Categorical/Discrete Imputation:** Missing values in `cylinders` and all other categorical columns (`engine`, `fuel`, `transmission`, `trim`, `body`, `doors`, `exterior_color`, `interior_color`) were filled with their respective **mode** (most frequent value).
2. **Feature Simplification:**
   - The `transmission` feature was simplified into four clean categories: `Automatic`, `Manual`, `CVT`, and `Other` to reduce feature space complexity.
3. **Categorical Encoding:**
   - **Label Encoding:** High-cardinality nominal columns (`name`, `model`, `engine`, `trim`, `exterior_color`, `interior_color`) were converted using `LabelEncoder`.
   - **One-Hot Encoding:** Low-cardinality nominal variables (`fuel`, `transmission`, `body`, `drivetrain`, `type`, `make`) were dummy-encoded using `pd.get_dummies` to avoid imposing an artificial order.
4. **Data Splitting:**
   - The dataset was divided into training (80%) and testing (20%) subsets using a `random_state=42` to ensure reproducibility.

---

## 📊 Model Performance Comparison

The following table summarizes the evaluation metrics across all five model implementations:

| Model | Hyperparameters | Train $R^2$ | Test $R^2$ | MAE ($) | RMSE ($) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | Default | N/A | **0.7240** | 7,369.34 | 9,181.46 |
| **Decision Tree (Vanilla)** | Unconstrained (`depth=None`) | **0.9994** | **0.4990** | 6,121.38 | 12,369.48 |
| **Decision Tree (Tuned)** | `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5` | **0.8632** | **0.7377** | 5,894.43 | 8,949.95 |
| **Random Forest (Tuned)** | `n_estimators=100`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5` | **0.8392** | **0.7195** | 6,026.53 | 9,255.65 |
| **Random Forest (Updated)** | `n_estimators=300` (Fully Grown) | **0.9724** | **0.7654** | **5,010.13** | **8,464.94** |

---

## 🧠 Theoretical Analysis & Discussion

### 1. Linear Regression
* **Performance:** Test $R^2 \approx 0.724$, MAE $\approx \$7,369$.
* **Theory:** Linear Regression operates under Ordinary Least Squares (OLS), assuming a linear relationship between features and the target variable. While it acts as a robust baseline and captures general linear relationships (e.g., newer cars and lower mileage lead to higher prices), it cannot inherently model non-linear interactions (e.g., how the combination of a specific premium brand and high-power engine exponentially boosts price) without manual interaction features.

### 2. Decision Tree (Vanilla vs. Tuned)
* **Vanilla Decision Tree (Overfitted):**
  * Train $R^2 = 0.9994$, Test $R^2 = 0.4990$, RMSE $= 12,369.48$.
  * **The Overfitting Problem:** A single unconstrained decision tree splits nodes recursively until all leaf nodes are pure (containing only one or a few samples). While this results in near-perfect training scores, the model captures random noise in the training set instead of the underlying distribution. This creates a high-variance model that generalizes poorly to unseen data, resulting in a low test $R^2$ and a massive RMSE.
* **Tuned Decision Tree (Regularized):**
  * Train $R^2 = 0.8632$, Test $R^2 = 0.7377$, RMSE $= 8,949.95$.
  * **The Regularization Solution:** By imposing structural constraints—restricting depth (`max_depth=10`), requiring at least 10 samples to split a node, and forcing at least 5 samples per leaf—we explicitly perform **early pruning**. This limits the model's complexity, forcing it to learn generalized rules rather than memorizing individual training rows. Consequently, training performance drops slightly (bias increases), but testing performance increases dramatically (variance decreases), leading to a much more generalizable and balanced model.

### 3. Random Forest (Tuned vs. Fully Grown)
* **Tuned Random Forest:**
  * Test $R^2 = 0.7195$, MAE $\approx \$6,026$.
  * Contrained parameters restricted individual trees too heavily in this ensemble configuration, leading to a slightly underfitted ensemble compared to the tuned single decision tree.
* **Fully Grown Random Forest (Updated):**
  * Train $R^2 = 0.9724$, Test $R^2 = 0.7654$, MAE $\approx \$5,010$, RMSE $\approx \$8,464$.
  * **The Power of Ensemble Bagging:** A Random Forest is an ensemble of decision trees trained using **Bootstrap Aggregating (Bagging)** and random feature selection. Each tree is trained on a random sample of data with replacement. 
  * **Why fully grown trees work in ensembles:** In Bagging, because we average the predictions of hundreds of unconstrained (high-variance, low-bias) trees, the overall variance is reduced by a factor of $N$ (number of trees), while the low bias is preserved. Therefore, we do not need to aggressively prune trees in a Random Forest. Allowing 300 deep, unconstrained trees to grow allows them to capture complex non-linear patterns, while the ensemble average handles the variance mitigation, resulting in the best overall test metrics.

---

## 📈 Key Learnings
1. **Ensemble over Single Estimators:** Ensembling via Random Forest outperforms single decision trees by combining many weak learners into a strong learner, yielding lower prediction error (MAE) and higher stability.
2. **Pruning is Crucial for Single Trees:** Single decision trees are exceptionally prone to overfitting. Without parameter tuning (limiting depth/leaf sizes), they are almost useless on test datasets.
3. **The Bias-Variance Tradeoff:** Restricting model complexity (regularization) reduces variance but increases bias. Finding the optimal sweep of hyperparameters is essential to maximize generalization.
