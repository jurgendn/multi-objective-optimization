import numpy as np

from poom.optimizers import Newton


def problem_1(x) -> float:
    return x[0]**2 + x[1] + 1


def problem_2(x) -> float:
    return x[0] - x[1]**2


optimizer = Newton()
optimizer.add_problem(problem_1)
optimizer.add_problem(problem_2)

out = optimizer.fit([0, 1])
print(out)
