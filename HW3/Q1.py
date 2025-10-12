import pandas as pd

# Data: (index, class, score)
data = [
    (1, 'T', 0.95),
    (2, 'T', 0.85),
    (3, 'N', 0.80),
    (4, 'T', 0.67),
    (5, 'T', 0.65),
    (6, 'T', 0.60),
    (7, 'N', 0.58),
    (8, 'N', 0.54),
    (9, 'T', 0.52),
    (10, 'N', 0.51),
    (11, 'T', 0.45),
    (12, 'N', 0.40),
    (13, 'N', 0.38),
    (14, 'N', 0.35),
    (15, 'N', 0.33),
    (16, 'N', 0.30),
    (17, 'T', 0.28),
    (18, 'N', 0.27),
    (19, 'N', 0.26),
    (20, 'N', 0.18),
]

# Create DataFrame
df = pd.DataFrame(data, columns=["index", "class", "score"])

# Thresholds to evaluate
thresholds = [i/10 for i in range(11)]

# Total positives and negatives
total_pos = sum(df['class'] == 'T')
total_neg = sum(df['class'] == 'N')

results = []

for th in thresholds:
    df['pred'] = df['score'] >= th

    TP = sum((df['pred']) & (df['class'] == 'T'))
    FP = sum((df['pred']) & (df['class'] == 'N'))
    FN = sum((~df['pred']) & (df['class'] == 'T'))
    TN = sum((~df['pred']) & (df['class'] == 'N'))

    TPR = TP / total_pos if total_pos > 0 else 0
    FPR = FP / total_neg if total_neg > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    results.append((f"≥ {th:.1f}", round(TPR, 4), round(FPR, 4), round(Precision, 4)))

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Threshold", "TPR/Recall", "FPR", "Precision"])

# Print as Markdown table
print(results_df)
