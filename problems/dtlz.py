from torch import cos, pi
import torch

n_variables = 4
n_objectives = 3

k = n_variables - n_objectives + 1


def g(x):
    x = torch.as_tensor(x)
    s = k
    for i in range(n_objectives, n_variables):
        s += (x[i] - 0.5)**2 - cos(20 * pi * (x[i] - 0.5))
    return 100 * s


def objective_generator(idx):
    def f(x):
        x = torch.as_tensor(x)
        s = 1 / 2
        for i in range(1, n_objectives - idx):
            s *= x[i]
        for i in range(n_objectives - idx, n_variables):
            s *= (1 - x[i])
        s *= (1 + g(x))
        return s

    return f


objs = [objective_generator(idx) for idx in range(n_objectives)]
