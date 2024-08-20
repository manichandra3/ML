import matplotlib.pyplot as plt
import numpy as np

from work.lab_utils_common import dlblue

plt.style.use('./deeplearning.mplstyle')


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best'):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1, )  #work with 1D or 1D y vectors
    neg = neg.reshape(-1, )

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c='red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def linear_regression(x, w, b):
    return np.dot(w, x) + b


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()


def loss_function(y, y_hat):
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw


def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]

    z = np.dot(X, w) + b
    f_wb = sigmoid(z)

    cost = -np.mean(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))

    return cost

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing

cost_history = list()
for i in range(1, 100):
    w = np.random.randn(2, 1)
    b = np.random.randn(1)
    cost = compute_cost(X_train, y_train, w, b)
    cost_history.append(cost)
    print('Cost at iteration {}: {}'.format(i, cost))
plt.plot(cost_history)
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.legend(['Cost'], loc='best')
plt.show()
