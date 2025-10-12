# Q4_ann.py
import numpy as np


# ---------------------------
# Activation functions
# ---------------------------
def leaky_relu(x, lam=0.1):
    return x if x > 0 else lam * x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------
# ANN forward function
# ---------------------------
def ann_forward(x):
    """
    Compute outputs of ANN for input x = [x1, x2, x3].
    Returns h1, h2, h3, h4, y
    """

    # Biases
    bias_hidden = [-5, -10, -6, 5]  # [h1, h2, h3, h4]
    bias_output = -1

    # Weights (only nonzero ones)
    w_x1_h1 = 2
    w_x1_h2 = 5
    w_x2_h2 = 1
    w_x2_h3 = 2
    w_x3_h3 = 3
    w_x3_h4 = 4

    w_h1_y = 1
    w_h2_y = 2
    w_h3_y = 2
    w_h4_y = 1

    # Hidden nodes
    z_h1 = x[0] * w_x1_h1 + bias_hidden[0]
    h1 = leaky_relu(z_h1)

    z_h2 = x[0] * w_x1_h2 + x[1] * w_x2_h2 + bias_hidden[1]
    h2 = leaky_relu(z_h2)

    z_h3 = x[1] * w_x2_h3 + x[2] * w_x3_h3 + bias_hidden[2]
    h3 = leaky_relu(z_h3)

    z_h4 = x[2] * w_x3_h4 + bias_hidden[3]
    h4 = leaky_relu(z_h4)

    # Output
    z_y = h1 * w_h1_y + h2 * w_h2_y + h3 * w_h3_y + h4 * w_h4_y + bias_output
    y = sigmoid(z_y)

    return h1, h2, h3, h4, y


def cross_entropy(y_pred, y_obs):
    # numerical stability
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -(y_obs * np.log(y_pred) + (1 - y_obs) * np.log(1 - y_pred))


def dL_dy(y, y_obs):
    return -y_obs / y + (1 - y_obs) / (1 - y)


# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    print("Output", ann_forward([1, 2, -1]))
    print("Output", ann_forward([1.5, 2.5, -0.5]))

    samples = [
        ([1, 2, -1], 0),    # sample 1
        ([1.5, 2.5, -0.5], 1)  # sample 2
    ]
    errors = []
    ce_losses = []
    for x, y_obs in samples:
        _, _, _, _, y_pred = ann_forward(x)
        errors.append((y_pred - y_obs) ** 2)
        print(f"Input: {x}, Predicted y: {y_pred:.4f}, Observed y: {y_obs}")
        y_pred = np.clip(y_pred, 1e-12, 1-1e-12)
        ce = -(y_obs * np.log(y_pred) + (1 - y_obs) * np.log(1 - y_pred))
        ce_losses.append(ce)
        print(f"Input: {x}, Predicted y: {y_pred:.4f}, Observed: {y_obs}, CE: {ce:.4f}")
        loss = cross_entropy(y_pred, y_obs)
        print("Q2.2 Cross-Entropy Loss:", loss)
        grad = dL_dy(y_pred, y_obs)
        print("dL/dy:", grad)
    mse = np.mean(errors)
    print("Q4.7 MSE:", mse)

    total_ce = np.sum(ce_losses)
    print("Q4.8 Total Cross-Entropy:", total_ce)

