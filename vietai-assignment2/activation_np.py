"""activation_np.py
This file provides activation functions for the NN 
Author: Phuong Hoang
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    #[TODO 1.1]
    #sigmoid(z) = 1/(1 + exp(−z))
    return 1/(1+np.exp(-x))


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    return a*(1-a)


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    #f(z) = max(0, z)
    return np.maximum(0, x)


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    relu_grad = 0
    if a > 0 : 1
    return reLU_grad


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    #tanh(z) = [exp(z) − exp (−z)]/[exp(z) + exp(−z)]
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    #return 2/(np.exp(a)+np.exp(−a))

    return 1 - a ** 2


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """

    e_Z = np.exp(x)
    output = e_Z / np.sum(e_Z, axis = 1, keepdims = True)
    return output


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    e_Z = np.exp(x - np.max(x, axis = 1, keepdims = True))
    output = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return output
