import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Load Data
# ----------------------------
file_path = "../Concrete_Data.csv"   # keep as-is if correct in your setup
data = pd.read_csv(file_path)

# Target and predictors
target_col = "Concrete compressive strength(MPa, megapascals) "
predictors = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)"
]

# Train/test split (same as Part A)
train = pd.concat([data.iloc[:501], data.iloc[631:]]).reset_index(drop=True)
test = data.iloc[501:631].reset_index(drop=True)

X_train = train[predictors]
X_test = test[predictors]
y_train = train[target_col]
y_test = test[target_col]

# ----------------------------
# Fit OLS Regression
# ----------------------------
X_train_const = sm.add_constant(X_train)   # add intercept term
model = sm.OLS(y_train, X_train_const).fit()

# ----------------------------
# Evaluate Performance
# ----------------------------
# Ensure intercept column is added consistently
X_test_const = sm.add_constant(X_test, has_constant="add")

y_train_pred = model.predict(X_train_const)
y_test_pred = model.predict(X_test_const)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("========================================")
print("Q1.1 Performance - Part B (OLS Regression)")
print("========================================")
print(f"MSE on training data: {train_mse:.4f}")
print(f"MSE on testing data: {test_mse:.4f}")
print(f"R² on training data: {train_r2:.4f}")
print(f"R² on testing data: {test_r2:.4f}")

# ----------------------------
# Statistical Analysis
# ----------------------------
print("\nModel Summary (statsmodels):")
print(model.summary())

# Optional: clean p-value display
print("\nP-values for each input feature:")
for feature, pval in model.pvalues.items():
    print(f"{feature:25s}: {pval:.4e}")
