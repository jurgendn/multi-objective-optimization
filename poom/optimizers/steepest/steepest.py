from typing import Callable, Iterable, List

import numpy as np
import torch
from scipy.optimize import minimize

from .utils import gradient


class LineSearch:

    def __init__(self,
                 objective_functions: List[Callable],
                 divider: float = 2,
                 step_size: float = 1,
                 sigma: float = 7e-1) -> None:
        self.divider = divider
        self.step_size = step_size
        self.objective_functions = objective_functions
        self.sigma = sigma

    def is_satisfy(self, x: Iterable, direction: Iterable, theta: float,
                   step_length) -> bool:
        for f in self.objective_functions:
            ref_values = f(x + step_length *
                           direction) - f(x) - self.sigma * step_length * theta
            if ref_values > 0:
                return False
        return True

    def armijo(self, x: Iterable, direction: Iterable, theta: float):
        step_length = self.step_size
        for _ in range(10):
            if self.is_satisfy(
                    x=x, direction=direction, theta=theta,
                    step_length=step_length) is True:
                break
            step_length /= 2
        return step_length

    def __call__(self, x, direction, theta):
        return self.armijo(x, direction, theta)


class SteepestDescent:

    def __init__(self,
                 objectives: List[Callable],
                 n_objectives: int = 2,
                 n_variables: int = 2,
                 divider: float = 2,
                 step_size: float = 1,
                 max_iteration: int = 100,
                 eps: float = 1) -> None:
        self.objectives = objectives
        self.n_objectives = n_objectives
        self.n_variables = n_variables
        self.eps = eps
        self.line_search = LineSearch(objective_functions=objectives,
                                      divider=divider,
                                      step_size=step_size,
                                      sigma=0.7)

    def nabla_F(self, x):
        nabla_F = np.zeros(
            (self.n_objectives, self.n_variables))  # (m, n) dimensional matrix
        for i, func in enumerate(self.objectives):
            nabla_F[i] = gradient(func, x)
        return nabla_F

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(np.dot(nabla_F, d)) + 0.5 * np.linalg.norm(d)**2

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * np.linalg.norm(d)**2

    def calc(self, x):
        res = []
        for func in self.objectives:
            res.append(func(x))
        return torch.Tensor(res)

    def fit(self, x):
        res = []
        for i in range(self.max_iteration):
            t = self.line_search.armijo(x=x, direction=d, theta=th)
            d = minimize(self.phi, x, args=(x, ))
            th = self.theta(d, x)
            x = x + t * d
            if abs(th) < self.eps:
                break
            y = self.calc(x)
            res.append(y)
        return res
