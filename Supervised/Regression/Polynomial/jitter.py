import random

import numpy as np
import matplotlib.pyplot as plt
from input import eval_2nd_degree, coeffs

hundred_xs = np.random.uniform(-10, 10, 100)
print(hundred_xs)


def eval_2nd_degree_jitter(coeffs, x, j):
    """
    Function to return the noisy output of evaluating a second degree polynomial,
    given a specific x value. Output values can be within [y−j,y+j].

    Args:
        coeffs: List containing the coefficients a,b, and c for the polynomial.
        x: The input x value to the polynomial.
        j: Jitter parameter, to introduce noise to output y.

    Returns:
        y: The corresponding jittered output y value for the second degree polynomial.

    """
    a = (coeffs[0] * (x * x))
    b = coeffs[1] * x
    c = coeffs[2]
    y = a + b + c
    print(y)

    interval = [y - j, y + j]
    interval_min = interval[0]
    interval_max = interval[1]
    print(f"Should get value in the range {interval_min} - {interval_max}")
    jit_val = random.random() * interval_max  # Generate a random number in range 0 to interval max

    while interval_min > jit_val:  # While the random jitter value is less than the interval min,
        jit_val = random.random() * interval_max  # it is not in the right range. Re-roll the generator until it
        # give a number greater than the interval min.

    return jit_val


# Test
x = 3
j = 4
eval_2nd_degree_jitter(coeffs, x, j)

x_y_pairs = []
for x in hundred_xs:
    y = eval_2nd_degree_jitter(coeffs, x, j)
    x_y_pairs.append((x, y))

xs = []
ys = []
for a, b in x_y_pairs:
    xs.append(a)
    ys.append(b)

plt.figure(figsize=(20, 10))
plt.plot(xs, ys, 'g+')
plt.title('Original data')
plt.show()

rand_coeffs = (random.randrange(-10, 10), random.randrange(-10, 10), random.randrange(-10, 10))

y_bar = eval_2nd_degree(rand_coeffs, hundred_xs)

plt.figure(figsize=(20, 10))
plt.plot(xs, ys, 'g+', label='original')
plt.plot(xs, y_bar, 'ro', label='prediction')
plt.title('Original data vs first prediction')
plt.legend(loc="lower right")
plt.show()


def loss_mse(ys, y_bar):
    """
    Calculates MSE loss.

    Args:
        ys: training data labels
        y_bar: prediction labels

    Returns: Calculated MSE loss.
    """
    return sum((ys - y_bar) * (ys - y_bar)) / len(ys)


initial_model_loss = loss_mse(ys, y_bar)


def calc_gradient_2nd_poly(rand_coeffs, hundred_xs, ys):
    """
    calculates the gradient for a second degree polynomial.

    Args:
        coeffs: a,b and c, for a 2nd degree polynomial [ y = ax^2 + bx + c ]
        inputs_x: x input datapoints
        outputs_y: actual y output points

    Returns: Calculated gradients for the 2nd degree polynomial, as a tuple of its parts for a,b,c respectively.

    """

    a_s = []
    b_s = []
    c_s = []

    y_bars = eval_2nd_degree(rand_coeffs, hundred_xs)

    for x, y, y_bar in list(
            zip(hundred_xs, ys, y_bars)):  # take tuple of (x datapoint, actual y label, predicted y label)
        x_squared = x ** 2
        partial_a = x_squared * (y - y_bar)
        a_s.append(partial_a)
        partial_b = x * (y - y_bar)
        b_s.append(partial_b)
        partial_c = (y - y_bar)
        c_s.append(partial_c)

    num = [i for i in y_bars]
    n = len(num)

    gradient_a = (-2 / n) * sum(a_s)
    gradient_b = (-2 / n) * sum(b_s)
    gradient_c = (-2 / n) * sum(c_s)
    return gradient_a, gradient_b, gradient_c  # return calculated gradients as a tuple of its 3 parts


calc_grad = calc_gradient_2nd_poly(rand_coeffs, hundred_xs, ys)

lr = 0.0001
a_new = rand_coeffs[0] - lr * calc_grad[0]
b_new = rand_coeffs[1] - lr * calc_grad[1]
c_new = rand_coeffs[2] - lr * calc_grad[2]

new_model_coeffs = (a_new, b_new, c_new)
print(f"New model coeffs: {new_model_coeffs}")
print("")

# update with these new coeffs:
new_y_bar = eval_2nd_degree(new_model_coeffs, hundred_xs)
updated_model_loss = loss_mse(ys, new_y_bar)

plt.figure(figsize=(20, 10))
plt.plot(xs, ys, 'g+', label='original model')
plt.plot(xs, y_bar, 'ro', label='first prediction')
plt.plot(xs, new_y_bar, 'b.', label='updated prediction')
plt.title('Original model vs 1st prediction vs updated prediction with lower loss')
plt.legend(loc="lower right")
plt.show()
