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

The following table summarizes the evaluation metrics across all model implementations:

| Model | Hyperparameters | Train $R^2$ | Test $R^2$ | MAE ($) | RMSE ($) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | Default | N/A | **0.7240** | 7,369.34 | 9,181.46 |
| **Decision Tree (Vanilla)** | Unconstrained (`depth=None`) | **0.9994** | **0.4990** | 6,121.38 | 12,369.48 |
| **Decision Tree (Tuned)** | `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5` | **0.8632** | **0.7377** | 5,894.43 | 8,949.95 |
| **Random Forest (Tuned)** | `n_estimators=100`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5` | **0.8392** | **0.7195** | 6,026.53 | 9,255.65 |
| **Random Forest (Updated)** | `n_estimators=300` (Fully Grown, All Features) | **0.9724** | **0.7654** | 5,010.13 | 8,464.94 |
| **Gradient Boosting** | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` | N/A | **0.7566** | 5,402.13 | 8,620.80 |
| **Random Forest (No 'name')** | `n_estimators=300` (Fully Grown) | N/A | **0.7589** | **4,973.47** | 8,580.85 |
| **Random Forest (No 'name', 'model', 'trim')** | `n_estimators=300` (Fully Grown) | **0.9680** | **0.7895** | 5,418.45 | **8,018.30** |
| **XGBoost (No 'name', 'model', 'trim')** | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05` | **0.9657** | **0.7778** | 5,476.51 | 8,238.01 |

---

## 🧠 Theoretical Analysis & Discussion

### 1. Linear Regression
* **Performance:** Test $R^2 \approx 0.7240$, MAE $\approx \$7,369$.
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
  * Constrained parameters restricted individual trees too heavily in this ensemble configuration, leading to a slightly underfitted ensemble compared to the tuned single decision tree.
* **Fully Grown Random Forest (Updated):**
  * Train $R^2 = 0.9724$, Test $R^2 = 0.7654$, MAE $\approx \$5,010$, RMSE $\approx \$8,464$.
  * **The Power of Ensemble Bagging:** A Random Forest is an ensemble of decision trees trained using **Bootstrap Aggregating (Bagging)** and random feature selection. Each tree is trained on a random sample of data with replacement. 
  * **Why fully grown trees work in ensembles:** In Bagging, because we average the predictions of hundreds of unconstrained (high-variance, low-bias) trees, the overall variance is reduced by a factor of $N$ (number of trees), while the low bias is preserved. Therefore, we do not need to aggressively prune trees in a Random Forest. Allowing 300 deep, unconstrained trees to grow allows them to capture complex non-linear patterns, while the ensemble average handles the variance mitigation, resulting in the best overall test metrics.

### 4. Gradient Boosting & XGBoost
* **Gradient Boosting Regressor (All Features):**
  * Test $R^2 = 0.7566$, MAE $\approx \$5,402$.
  * Gradient Boosting builds trees sequentially (boosting) rather than independently (bagging). Each new tree is trained to correct the residual errors of the existing ensemble. It uses shallow trees (max depth = 4) to prevent overfitting.
* **XGBoost (No 'name', 'model', 'trim'):**
  * Train $R^2 = 0.9657$, Test $R^2 = 0.7778$, MAE $\approx \$5,477$, RMSE $= 8,238.01$.
  * XGBoost (Extreme Gradient Boosting) is an optimized, highly efficient implementation of gradient boosting. It utilizes advanced regularization ($L_1$/$L_2$), parallel processing, and handling of sparse patterns. By dropping high-cardinality features, XGBoost generalizes exceptionally well, achieving a Test $R^2$ of $0.7778$.

### 5. High-Cardinality Feature Engineering: The Impact of Dropping 'name', 'model', and 'trim'
* **The Pitfall of Label Encoding:** High-cardinality categorical variables like `name` (the specific car listing title), `model`, and `trim` contain a very large number of unique categories. Applying `LabelEncoder` encodes these categories as arbitrary integers (e.g. 0 to $K-1$). This introduces a spurious ordinal relationship (e.g., model 15 is somehow "greater than" model 2).
* **Ensemble Memorization vs. Generalization:**
  * When using all features, a Random Forest or Decision Tree splits heavily on these arbitrary integers to identify specific cars, resulting in low training error (memorization) but higher generalization error.
  * In the fully grown Random Forest with all features, `name` and `model` had feature importances of **14.6%** and **14.3%** respectively—ranking among the highest. This indicates the model heavily relied on these noisy identifier columns.
* **Why Dropping Them Improved the Model:**
  * **Random Forest (No 'name'):** Yields the lowest MAE of **\$4,973.47** (Test $R^2 = 0.7589$). Dropping the listing `name` immediately prevents the model from memorizing specific listings.
  * **Random Forest (No 'name', 'model', 'trim'):** Achieves the best overall Test $R^2$ of **0.7895** and the lowest RMSE of **8,018.30**. By removing all three high-cardinality identifier fields, we force the model to base its predictions purely on physical characteristics: `year`, `mileage`, `cylinders`, `engine`, `transmission`, and `make`. This prevents overfitting, drastically reduces the magnitude of large errors (lowering RMSE by over 400 points compared to the baseline), and delivers the most stable and generalizable model.

---

## 📈 Key Learnings & Takeaways
1. **The High-Cardinality Encoding Pitfall:** Encoding high-cardinality features like specific names or models using `LabelEncoder` causes trees to overfit by treating arbitrary integer IDs as ordered splits. In tree-based models, it is often better to drop these identifiers if they don't represent a broad group, or use target encoding/one-hot encoding.
2. **Dropping Features Can Improve Performance:** Sometimes, "less is more." Dropping high-cardinality identifiers forced the ensemble models to learn generalizable rules from physical features (`year`, `mileage`, `cylinders`), raising the Test $R^2$ from **0.7654** to **0.7895** and reducing the RMSE.
3. **RMSE vs. MAE Tradeoffs:** Dropping `name`, `model`, and `trim` slightly increased the Mean Absolute Error (MAE) from \$5,010 to \$5,418, but significantly reduced the Root Mean Squared Error (RMSE) from 8,464 to 8,018. This demonstrates a massive reduction in large prediction outliers (the model makes fewer "wildly wrong" predictions).
4. **Ensemble Methods rule for Complex Tabular Data:** Random Forest (bagging) and XGBoost (boosting) outperform simpler models like Linear Regression and single Decision Trees by effectively managing variance and non-linear interactions.
