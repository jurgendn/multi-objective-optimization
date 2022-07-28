n_variables = 2
n_objectives = 2


def f(x):
    return (x[0] - 1)**2 + 3 * (x[1] - 10)**2


def g(x):
    return 2 * (x[1] - 10)**4 + (x[0] - 1)**2


objs = [f, g]