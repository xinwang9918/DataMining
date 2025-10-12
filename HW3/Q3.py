# Q3_wine_classification.py
import pandas as pd

# ---------------------------
# Dataset (Wine Samples)
# ---------------------------
data = [
    (1, 0.50, 2.0, "Red", 4),
    (2, 0.00, 1.9, "Red", 1),
    (3, 0.00, 1.8, "Red", 2),
    (4, 0.06, 1.6, "Red", 2),
    (5, 0.65, 1.2, "Rose", 8),
    (6, 0.37, 1.2, "Rose", 3),
    (7, 0.40, 1.5, "White", 8),
    (8, 0.62, 19.3, "White", 4),
    (9, 0.38, 1.5, "White", 9),
    (10, 0.04, 1.1, "White", 6),
]

df = pd.DataFrame(data, columns=["ID", "CitricAcid", "Sugar", "Color", "TrueQuality"])

# One-hot encoding for colors
df["Red"]   = (df["Color"] == "Red").astype(int)
df["Rose"]  = (df["Color"] == "Rose").astype(int)
df["White"] = (df["Color"] == "White").astype(int)

# ---------------------------
# Model formula from prompt
# ---------------------------
df["PredictedQuality"] = (
        1.81
        + 7.41 * df["CitricAcid"]
        - 0.35 * df["Sugar"]
        + 0.04 * df["Red"]
        + 0.33 * df["Rose"]
        + 4.31 * df["White"]
)

# Ground truth: high quality wines are IDs 5, 7, 9
df["TrueHighQuality"] = df["ID"].isin([5, 7, 9]).astype(int)

# ---------------------------
# Helper: compute confusion counts for a threshold on PredictedQuality
# ---------------------------
def confusion_counts(threshold: float):
    pred_pos = (df["PredictedQuality"] > threshold).astype(int)  # strictly "greater than 4.5"
    true_pos = df["TrueHighQuality"]

    TP = int(((pred_pos == 1) & (true_pos == 1)).sum())
    FP = int(((pred_pos == 1) & (true_pos == 0)).sum())
    FN = int(((pred_pos == 0) & (true_pos == 1)).sum())
    TN = int(((pred_pos == 0) & (true_pos == 0)).sum())
    return TP, FP, FN, TN

# ---------------------------
# Threshold = 4.5 (Q3.1, Q3.2, Q3.3, Q3.4)
# ---------------------------
TP, FP, FN, TN = confusion_counts(4.5)

precision = TP / (TP + FP) if (TP + FP) else 0.0          # Q3.1
recall    = TP / (TP + FN) if (TP + FN) else 0.0           # Q3.2
fpr       = FP / (FP + TN) if (FP + TN) else 0.0           # Q3.3 (correct)
accuracy_4_5 = (TP + TN) / (TP + FP + FN + TN)             # Q3.4

print("Q3.1 Precision:", precision)
print("Q3.2 Recall:", recall)
print("Q3.3 FPR:", fpr)                   # <- FPR = FP / (FP + TN)
print("Q3.4 Accuracy (threshold 4.5):", accuracy_4_5)
print(f"Confusion @4.5 -> TP:{TP}, FP:{FP}, FN:{FN}, TN:{TN}")

# ---------------------------
# Threshold = 8.5 (Q3.5)
# ---------------------------
TP2, FP2, FN2, TN2 = confusion_counts(8.5)
accuracy_8_5 = (TP2 + TN2) / (TP2 + FP2 + FN2 + TN2)

print("Q3.5 Accuracy (threshold 8.5):", accuracy_8_5)
print(f"Confusion @8.5 -> TP:{TP2}, FP:{FP2}, FN:{FN2}, TN:{TN2}")
