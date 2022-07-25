from typing import Callable

import torch
from scipy.optimize import approx_fprime
from torch import Tensor
from torch.autograd import functional as tf


def gradient(f: Callable, x):
    return approx_fprime(x, f, epsilon=1e-5)


def hessian(f: Callable, x):
    x = Tensor(x)
    with torch.no_grad():
        h = tf.hessian(f, x)
    return h


def get_approx(func: Callable, x):
    g = gradient(func, x)
    h = hessian(func, x)
    return g, h
