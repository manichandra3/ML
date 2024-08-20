import math
import numpy as np
import matplotlib.pyplot as plt

coeffs = [2, -5, 4]


def eval_2nd_degree(coeffs, x):
    """
    Function to return the output of evaluating a second degree polynomial,
    given a specific x value.

    Args:
        coeffs: List containing the coefficients a,b, and c for the polynomial.
        x: The input x value to the polynomial.

    Returns:
        y: The corresponding output y value for the second degree polynomial.

    """
    a = (coeffs[0] * (x * x))
    b = coeffs[1] * x
    c = coeffs[2]
    y = a + b + c
    return y
