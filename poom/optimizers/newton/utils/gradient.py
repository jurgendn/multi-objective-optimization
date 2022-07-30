from typing import Callable

import torch
from scipy.optimize import approx_fprime
from torch.autograd import functional as tf


def gradient(f: Callable, x):
    x = torch.as_tensor(x).float()
    grad = tf.jacobian(f, x)
    return grad


def hessian(f: Callable, x):
    x = torch.as_tensor(x).float()
    h = tf.hessian(f, x)
    return h


def get_approx(func: Callable, x):
    g = gradient(func, x)
    h = hessian(func, x)
    return g.numpy(), h.numpy()
