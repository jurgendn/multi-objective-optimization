from math import cos as mcos
from math import sin as msin

import torch
from torch import cos, sin

n_variables = 2
n_objectives = 2


def f1(x):
    x = torch.as_tensor(x)
    A1 = 0.5 * msin(1) - 2 * mcos(1) + msin(2) - 1.5 * mcos(2)
    B1 = 0.5 * sin(x[0]) - 2 * cos(x[0]) + sin(x[1]) - 1.5 * cos(x[1])
    A2 = 1.5 * msin(1) - mcos(1) + 2 * msin(2) - 0.5 * mcos(2)
    B2 = 1.5 * sin(x[0]) - cos(x[0]) + 2 * sin(x[1]) - 0.5 * cos(x[1])
    return (1 + (A1 - B1)**2 + (A2 - B2)**2)


def f2(x):
    x = torch.as_tensor(x)
    return ((x[0] + 3)**2) + (x[1] + 1)**2


def constraint_1(x):
    status = (-2 <= x[0]) and (x[0] <= 2)
    return status


def constraint_2(x):
    status = (-2 <= x[1]) and (x[1] <= 2)
    return status


def constraints(x):
    return constraint_1(x) * constraint_2(x)


objs = [f1, f2]
