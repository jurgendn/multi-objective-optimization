n_variables = 3
n_objectives = 3


def f(x):
    return x[0]**2 + 3 * (x[1] - 1)**2 + (x[2] - 2)**2


def g(x):
    return 2 * (x[0] - 1)**2 + x[1]**2


def h(x):
    return 2 * (x[0] - 1)**2 + x[2]**2


constraints = []


def constraint_1(x):
    first = (-1 < x[0]) and (x[0] < 1)
    return first


def constraint_2(x):
    second = (-1 < x[1]) and (x[1] < 1)
    return second


def constraints(x):
    return constraint_1(x) and constraint_2(x)


objs = [f, g, h]