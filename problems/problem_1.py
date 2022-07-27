import torch

n_variables = 30
n_objectives = 2


def f1(x):
    return x[0]


def f2(x):
    n = x.shape[0]
    g = 1 + 9 / (n - 1) * x[1:].sum()
    h = 1 - torch.sqrt(x[0] / g)
    return g * h


objs = [f1, f2]
