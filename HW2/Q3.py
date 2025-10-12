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

    dm = (2/n) * np.sum(error * x)
    db = (2/n) * np.sum(error)

    m_new = m - alpha * dm
    b_new = b - alpha * db
    return m_new, b_new


# Q3.1
m, b = 1, 1
m_new, b_new = gradient_descent_step([3], [4], m, b, alpha=0.1)
print("Q3.1:", m_new, b_new)

# Q3.2
m, b = 1, 1
m_new, b_new = gradient_descent_step([3], [3], m, b, alpha=0.1)
print("Q3.2:", m_new, b_new)

# Q3.3
m, b = 1, 1
x = [3, 4, 10]
y = [3, 2, 8]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.3:", m_new, b_new)

# Q3.4
m, b = 1, 1
x = [3, 4, 10, 3, 11]
y = [3, 2, 8, 4, 5]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.4:", m_new, b_new)

# Q3.5
m, b = 1, 1
x = [540]
y = [80]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.5:", m_new, b_new)

# Q3.6
m, b = 1, 1
x = [540, 332, 266, 380, 199][:3]
y = [80, 40, 47, 44, 38][:3]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.6:", m_new, b_new)

# Q3.7
m, b = 1, 1
x = [540, 332, 266, 380, 199]
y = [80, 40, 47, 44, 38]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.7:", m_new, b_new)

# Q3.8
m, b = 1, -350
x = [540, 332, 266, 380, 199]
y = [80, 40, 47, 44, 38]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.8:", m_new, b_new)

# Q3.9
m, b = 1, 1
x = [540, 332, 266, 380, 199]
y = [80, 40, 47, 44, 38]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.0001)
print("Q3.9:", m_new, b_new)

# Q3.10
m, b = 1, 1
x = [1, 0.39, 0.2, 0.53, 0]
y = [80, 40, 47, 44, 38]
m_new, b_new = gradient_descent_step(x, y, m, b, alpha=0.1)
print("Q3.10:", m_new, b_new)
