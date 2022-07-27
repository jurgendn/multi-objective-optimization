n_variables = 4
n_objectives = 3


def f(x):
    return x[0]**2 + 3 * (x[1] - 1)**2 + (x[2] - 2)**2


def g(x):
    return 2 * (x[0] - 1)**2 + x[1]**2


def h(x):
    return 2 * (x[0] - 1)**2 + x[2]**2


objs = [f, g, h]
