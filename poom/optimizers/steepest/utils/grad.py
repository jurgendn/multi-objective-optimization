from typing import Callable

import torch
from scipy.optimize import approx_fprime
from torch.autograd import functional as tf


def gradient(f: Callable, x):
    f_prime = approx_fprime(x, f, epsilon=1e-7)
    return torch.from_numpy(f_prime).float()


def hessian(f: Callable, x):
    x = torch.as_tensor(x).float()
    h = tf.hessian(f, x)
    return h


def get_approx(func: Callable, x):
    g = gradient(func, x)
    h = hessian(func, x)
    return g, h
