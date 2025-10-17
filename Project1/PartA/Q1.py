import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# Gradient Descent Functions
# ----------------------------
def gradient_descent(x, y, lr=0.01, epochs=1000):
    m, b = 0.0, 0.0
    n = len(y)
    for _ in range(epochs):
        y_pred = m * x + b
        error = y_pred - y
        dm = (2 / n) * np.dot(error, x)
        db = (2 / n) * np.sum(error)
        m -= lr * dm
        b -= lr * db
    return m, b


def evaluate(x, y, m, b):
    y_pred = m * x + b
    mse = np.mean((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return mse, r2


# ----------------------------
# Load Data
# ----------------------------
file_path = "../Concrete_Data.csv"  # 修改成你的数据文件名
data = pd.read_csv(file_path)

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

# Train/test split: rows 0-500 and 631-end for train, 501-630 for test
train = pd.concat([data.iloc[:501], data.iloc[631:]]).reset_index(drop=True)
test = data.iloc[501:631].reset_index(drop=True)

y_train = train[target_col].values
y_test = test[target_col].values

# ----------------------------
# Q1.1 Set 1 (Normalized Predictors)
# ----------------------------
print("\n" + "=" * 40)
print("Q1.1 Set 1 - Normalized Predictors")
print("=" * 40)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train[predictors])
X_test_scaled = scaler.transform(test[predictors])

for i, col in enumerate(predictors):
    x_train = X_train_scaled[:, i]
    x_test = X_test_scaled[:, i]

    m, b = gradient_descent(x_train, y_train, lr=0.05, epochs=5000)
    train_mse, train_r2 = evaluate(x_train, y_train, m, b)
    test_mse, test_r2 = evaluate(x_test, y_test, m, b)

    print(f"\n{col} as predictor")
    print(f"m and b values: {m:.4f}, {b:.4f}")
    print(f"MSE on training data: {train_mse:.4f}")
    print(f"Variance Explained (R²) on training data: {train_r2:.4f}")
    print(f"MSE on testing data: {test_mse:.4f}")
    print(f"Variance Explained (R²) on testing data: {test_r2:.4f}")

# ----------------------------
# Q1.2 Set 2 (Raw Predictors - Fixed)
# ----------------------------
print("\n" + "=" * 40)
print("Q1.2 Set 2 - Raw Predictors (Refined)")
print("=" * 40)

X_train_raw = train[predictors].values
X_test_raw = test[predictors].values

# 根据每个 predictor 的数值范围选择不同的学习率
learning_rates = {
    "Cement (component 1)(kg in a m^3 mixture)": 1e-7,
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": 1e-7,
    "Fly Ash (component 3)(kg in a m^3 mixture)": 1e-7,
    "Water  (component 4)(kg in a m^3 mixture)": 1e-7,
    "Superplasticizer (component 5)(kg in a m^3 mixture)": 1e-6,
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": 1e-7,
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": 1e-7,
    "Age (day)": 1e-5
}

for i, col in enumerate(predictors):
    x_train = X_train_raw[:, i]
    x_test = X_test_raw[:, i]

    # 中心化 (NOT standardization)
    x_mean = np.mean(x_train)
    x_train_centered = x_train - x_mean
    x_test_centered = x_test - x_mean

    lr = learning_rates[col]
    m, b = gradient_descent(x_train_centered, y_train, lr=lr, epochs=50000)

    train_mse, train_r2 = evaluate(x_train_centered, y_train, m, b)
    test_mse, test_r2 = evaluate(x_test_centered, y_test, m, b)

    print(f"\n{col} as predictor")
    print(f"m and b values: {m:.6f}, {b:.6f}")
    print(f"MSE on training data: {train_mse:.4f}")
    print(f"Variance Explained (R²) on training data: {train_r2:.4f}")
    print(f"MSE on testing data: {test_mse:.4f}")
    print(f"Variance Explained (R²) on testing data: {test_r2:.4f}")

