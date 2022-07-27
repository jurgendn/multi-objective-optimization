n_variables = 2
n_objectives = 2


def f1(x):
    return x[0]**2 + 3 * (x[1] - 1)**2


def f2(x):
    return 2 * (x[0] - 1)**2 + x[1]**2


objs = [f1, f2]