import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
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

# True labels (1 for T, 0 for N)
y_true = [1 if c == 'T' else 0 for c in df['class']]
# Scores
y_scores = df['score']

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

# Plot PR curve
plt.figure(figsize=(6,6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()

# Save PR curve
plt.savefig("pr_curve.png", dpi=300)   # Save as PNG
plt.savefig("pr_curve.pdf")           # Save as PDF
plt.close()
