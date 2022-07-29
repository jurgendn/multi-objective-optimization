n_variables = 3
n_objectives = 2

from math import sqrt

import torch


def f(x):
    x = torch.as_tensor(x)
    s = (x[0] - 1 / sqrt(3))**2 + (x[1] - 1 / sqrt(3))**2 + (x[2] -
                                                             1 / sqrt(3))**2
    s = torch.exp(-s)
    return 1 - s


def g(x):
    x = torch.as_tensor(x)
    s = (x[0] + 1 / sqrt(3))**2 + (x[1] + 1 / sqrt(3))**2 + (x[2] +
                                                             1 / sqrt(3))**2
    s = torch.exp(-s)
    return 1 - s


objs = [f, g]
