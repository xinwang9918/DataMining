import pandas as pd
import numpy as np
import statsmodels.api as sm

# ----------------------------
# Load Data
# ----------------------------
file_path = "../Concrete_Data.csv"   # adjust if needed
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
y_train = train[target_col]

# ----------------------------
# Log transform predictors (add 1 first to avoid log(0))
# ----------------------------
X_train_log = np.log(X_train + 1)
X_train_log = pd.DataFrame(X_train_log, columns=predictors)

# ----------------------------
# Fit OLS Regression on log-transformed predictors
# ----------------------------
X_train_const = sm.add_constant(X_train_log)
model_log = sm.OLS(y_train, X_train_const).fit()

# ----------------------------
# Print p-values
# ----------------------------
print("P-values for log-transformed predictors:")
for feature, pval in model_log.pvalues.items():
    print(f"{feature:25s}: {pval:.4e}")

# Optional: show summary
print("\nModel Summary (log-transformed predictors):")
print(model_log.summary())
print("\nP-values for each input feature:")
for feature, pval in model_log.pvalues.items():
    print(f"{feature:25s}: {pval:.4e}")