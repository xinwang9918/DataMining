import numpy as np

# -----------------------------
# CNN + Dense Network Functions
# -----------------------------

def cnn_forward(image, true_label=None):
    """
    Runs a 6x6 binary image through the CNN pipeline and returns outputs.
    true_label: optional, can be "O", "X", or "?" for cross-entropy loss.
    """

    # ----- Convolution -----
    kernel = np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0]
    ], dtype=int)

    k = 3
    feature = np.zeros((4,4), dtype=int)
    for i in range(4):
        for j in range(4):
            feature[i,j] = np.sum(image[i:i+k, j:j+k] * kernel)

    # ----- Bias b1 and ReLU -----
    biased = feature - 2
    cleaned = np.maximum(0, biased)

    # ----- Max Pooling 2x2 -----
    pool_UL = np.max(cleaned[0:2, 0:2])
    pool_UR = np.max(cleaned[0:2, 2:4])
    pool_LL = np.max(cleaned[2:4, 0:2])
    pool_LR = np.max(cleaned[2:4, 2:4])
    pooled = np.array([pool_UL, pool_UR, pool_LL, pool_LR], dtype=float)

    # ----- Hidden node -----
    w = np.array([-0.8, -0.07, 0.2, 0.17])
    b2 = 0.97
    h_pre = pooled @ w + b2
    h = max(0.0, h_pre)  # ReLU

    # ----- Output nodes -----
    O = -1.33*h + 1.45
    X =  1.33*h - 0.45
    Q =  1.00*h + 0.50

    # ----- SoftMax -----
    logits = np.array([O, X, Q])
    exps = np.exp(logits)
    probs = exps / np.sum(exps)

    results = {
        "feature_map": feature,
        "biased_map": biased,
        "cleaned_map": cleaned,
        "pooled": pooled,
        "hidden_pre": h_pre,
        "hidden": h,
        "logits": {"O": O, "X": X, "?": Q},
        "probs": {"O": probs[0], "X": probs[1], "?": probs[2]}
    }

    # ----- Cross-Entropy Loss -----
    if true_label is not None:
        idx = {"O": 0, "X": 1, "?": 2}[true_label]
        results["cross_entropy"] = -np.log(probs[idx])

    return results


# -----------------------------
# Example Usage
# -----------------------------

# Handwritten O example (from earlier Q1.1)
O_img = np.array([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0]
], dtype=int)

# Run through CNN
results = cnn_forward(O_img, true_label="?")

# Print key outputs
print("Feature map (before bias):")
print(results["feature_map"])
print("\nCleaned map (after ReLU):")
print(results["cleaned_map"])
print("\nPooled values:", results["pooled"])
print("\nHidden node value:", results["hidden"])
print("\nLogits:", results["logits"])
print("\nProbabilities:", results["probs"])
print("\nCross Entropy Loss (true=1):", results["cross_entropy"])
