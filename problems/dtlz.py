from math import cos, pi

n_variables = 4
n_objectives = 3

k = n_variables - n_objectives + 1


def g(x):
    s = k
    for i in range(k, n_objectives):
        s += (x[i] - 0.5)**2 - cos(20 * pi * (x[i] - 0.5))
    return 100 * s


def objective_generator(idx):

    def f(x):
        s = 1 / 2
        for i in range(1, n_objectives - idx):
            s *= x[i]
            s *= (1 - x[n_variables - i - 1])
        s *= (1 + g(x))
        return s

    return f


objs = [objective_generator(idx) for idx in range(n_objectives)]
