from typing import Callable

import torch
from scipy.optimize import approx_fprime
from torch import Tensor
from torch.autograd import functional as tf


def gradient(f: Callable, x):
    with torch.no_grad():
        f_prime = approx_fprime(x, f, epsilon=1e-5)
    return f_prime


def hessian(f: Callable, x):
    x = torch.as_tensor(x).float()
    with torch.no_grad():
        h = tf.hessian(f, x)
    return h


def get_approx(func: Callable, x):
    g = gradient(func, x)
    h = hessian(func, x)
    return g, h
