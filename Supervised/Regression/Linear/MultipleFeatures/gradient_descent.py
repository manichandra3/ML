import copy
import math

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('deeplearning.mplstyle')
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
# for demonstration w and b are initially loaded with some selected values that are near optimal.
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
    :param x: Shape (n) example with multiple features
    :param w: Shape (n) model parameters
    :param b: model parameter
    :return : p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += w[i] * x[i]
    p += b
    return p


def predict(x, w, b):
    """
    predict using vectorization

    :param x: Shape (n) example with multiple features
    :param w: Shape (n) model parameters
    :param b: model parameter
    :return : p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p


def compute_cost(X, y, w, b):
    """
    compute the cost function

    :param X: Shape (n) example with multiple features
    :param y: Shape (n) example with multiple labels
    :param w: Shape (n) model parameters
    :param b: model parameter
    :return: cost (scalar)
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar
        cost += np.square(f_wb_i - y[i])  # scalar
    cost = cost / 2 * m  # scalar
    return cost


# def compute_gradient(X, y, w, b):
#     """
#     compute the gradient function
#
#     :param X: Shape (n) example with multiple features
#     :param y: Shape (n) example with multiple labels
#     :param w: Shape (n) model parameters
#     :param b: model parameter
#     :return: gradient (scalar)
#     """
#     m, n = X.shape  # number of examples, number of features
#     dj_dw = np.zeros_like((n,))
#     dj_db = 0
#
#     for i in range(m):
#         err = y[i] - predict(X[i], w, b)
#         for j in range(n):
#             dj_dw += err * X[i][j]
#         dj_db += err
#     dj_dw = dj_dw / m
#     dj_db = dj_db / m
#     return dj_db, dj_dw


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iter):
    """
    gradient descent using multiple linear regression

    :param X: Shape (n) example with multiple features
    :param y: Shape (n) example with multiple labels
    :param w_in: Shape (n) model parameters
    :param b_in: model parameter
    :param cost_function: loss function
    :param gradient_function: gradient function
    :param alpha: learning rate
    :param num_iter: number of iterations
    :return: w_final and b_final after optimization
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iter):
        # calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  # None

        # update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  # None
        b = b - alpha * dj_db  # None

        # save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# get a row from our training data
x_vec = X_train[0, :]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# Compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()
