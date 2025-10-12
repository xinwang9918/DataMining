import numpy as np


def gradient_descent_step(x, y, m, b, alpha):
    """
    Perform one step of gradient descent for univariate linear regression.
    x, y can be scalars or lists/arrays.
    """
    # x = np.array(x, dtype=float)
    # y = np.array(y, dtype=float)
    # y_pred = m * x + b
    # error = y - y_pred
    # n = len(x)
    #
    # dm = -2 * np.sum(x * error)
    # db = -2 * np.sum(error)
    #
    # m_new = m - (alpha * dm /n)
    # b_new = b - (alpha * db /n)
    #
    # return m_new, b_new

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    y_pred = m * x + b
    error = y_pred - y
    n = len(x)

    dm = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)

    m_new = m - alpha * dm
    b_new = b - alpha * db
    return m_new, b_new


print(gradient_descent_step([250], [14], 0.1, -10, 0.1))
