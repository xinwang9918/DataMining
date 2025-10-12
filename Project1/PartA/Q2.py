import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# Utils (LossB: J = (1/n) * sum e^2, gradients have ×2)
# ----------------------------
def one_step_update_lossB(X, y, m, b, alpha, reduction="mean"):
    """
    X: (n,d), y: (n,), m: (d,), b: scalar
    LossB: J = (1/n) * sum (yhat - y)^2  -> grad_m = (2/n) X^T e, grad_b = (2/n) sum e
    reduction: "mean" (默认) 或 "sum"（若评测特别用sum，再切换）
    """
    yhat = X @ m + b
    e = yhat - y
    n = len(y)
    scale = (1.0 / n) if reduction == "mean" else 1.0
    grad_m = 2.0 * scale * (X.T @ e)
    grad_b = 2.0 * scale * e.sum()
    m_new = m - alpha * grad_m
    b_new = b - alpha * grad_b
    return m_new, b_new


def evaluate(X, y, m, b):
    yhat = X @ m + b
    mse = np.mean((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, r2


# 格式化输出 m 值
def format_m(m):
    return ", ".join([f"{val:.6f}" for val in m])


# ----------------------------
# Q2.1 Code test 1 (LossB)
# ----------------------------
print("\n" + "=" * 40)
print("Q2.1 Code Test 1 (LossB, mean)")
print("=" * 40)
x = np.array([[3.0, 4.0, 5.0]])  # (1,3)
y = np.array([4.0])  # (1,)
m = np.array([1.0, 1.0, 1.0]);
b = 1.0
alpha = 0.1
m, b = one_step_update_lossB(x, y, m, b, alpha, reduction="mean")
print(f"New m_1: {m[0]:.2f}")
print(f"New m_2: {m[1]:.2f}")
print(f"New m_3: {m[2]:.2f}")
print(f"New b:   {b:.2f}")
# 期望：m=[-4.40, -6.20, -8.00], b=-0.80

# ----------------------------
# Q2.2 Code test 2 (LossB)
# ----------------------------
print("\n" + "=" * 40)
print("Q2.2 Code Test 2 (LossB, mean)")
print("=" * 40)
X = np.array([[3, 4, 4],
              [4, 2, 1],
              [10, 2, 5],
              [3, 4, 5],
              [11, 1, 1]], dtype=float)
y = np.array([3, 2, 8, 4, 5], dtype=float)
m = np.array([1.0, 1.0, 1.0]);
b = 1.0
alpha = 0.1
m, b = one_step_update_lossB(X, y, m, b, alpha, reduction="mean")
print(f"New m_1: {m[0]:.2f}")
print(f"New m_2: {m[1]:.2f}")
print(f"New m_3: {m[2]:.2f}")
print(f"New b:   {b:.2f}")
# 期望：m≈[-10.08, -3.52, -4.84], b≈-0.72

# ----------------------------
# 数据加载（与你的数据集列名一致）
# ----------------------------
file_path = "../Concrete_Data.csv"  # 按需要修改
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

# 训练/测试划分：train = rows [0..500] ∪ [631..end]； test = rows [501..630]
train = pd.concat([data.iloc[:501], data.iloc[631:]]).reset_index(drop=True)
test = data.iloc[501:631].reset_index(drop=True)
X_train_raw = train[predictors].values
X_test_raw = test[predictors].values
y_train = train[target_col].values
y_test = test[target_col].values

# ----------------------------
# Q2.3 Set 1（Normalized predictors, raw y）— LossB
# ----------------------------
print("\n" + "=" * 40)
print("Q2.3 Set 1 - Normalized Predictors (LossB)")
print("=" * 40)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


# 训练（多轮 GD）
def train_multi_lossB(X, y, lr=0.05, epochs=5000):
    n, d = X.shape
    m = np.ones(d)  # 作业要求可从 1 开始（与上面 code test 一致风格）
    b = 1.0
    for _ in range(epochs):
        m, b = one_step_update_lossB(X, y, m, b, lr, reduction="mean")
    return m, b


m, b = train_multi_lossB(X_train, y_train, lr=0.05, epochs=5000)
train_mse, train_r2 = evaluate(X_train, y_train, m, b)
test_mse, test_r2 = evaluate(X_test, y_test, m, b)

print(f"m values: {format_m(np.round(m, 6))}")
print(f"b value:  {b:.6f}")
print(f"MSE on training data: {train_mse:.4f}")
print(f"Variance Explained (R²) on training data: {train_r2:.4f}")
print(f"MSE on testing data: {test_mse:.4f}")
print(f"Variance Explained (R²) on testing data: {test_r2:.4f}")

# ----------------------------
# Q2.4 Set 2（Raw predictors + Raw y）— LossB
# ----------------------------
print("\n" + "=" * 40)
print("Q2.4 Set 2 - Raw Predictors (LossB)")
print("=" * 40)
# 原始尺度差异很大，GD 要用很小的 lr
m, b = train_multi_lossB(X_train_raw, y_train, lr=1e-7, epochs=15000)
train_mse, train_r2 = evaluate(X_train_raw, y_train, m, b)
test_mse, test_r2 = evaluate(X_test_raw, y_test, m, b)

print(f"m values: {format_m(np.round(m, 6))}")
print(f"b value:  {b:.6f}")
print(f"MSE on training data: {train_mse:.4f}")
print(f"Variance Explained (R²) on training data: {train_r2:.4f}")
print(f"MSE on testing data: {test_mse:.4f}")
print(f"Variance Explained (R²) on testing data: {test_r2:.4f}")
