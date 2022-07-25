from typing import Callable, Iterable, List

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from tqdm.auto import tqdm

from .utils import get_approx
from .utils.report import ResultReporter


class NewtonDirection:
    def __init__(self, func_list: List[Callable]) -> None:
        self.objective = lambda x: x[-1]
        self.n_objective = len(func_list)
        self.func_list = func_list

    def __make_constraint(self, func: Callable, x):
        g, h = get_approx(func, x)

        def constraints(s):
            return np.transpose(g).dot(
                s[:-1]) + (np.transpose(s[:-1]).dot(h).dot(s[:-1])) / 2 - s[-1]

        return constraints

    def make_constraints(self, x):
        lhs = []
        for func in self.func_list:
            f = self.__make_constraint(func, x)
            constraint = NonlinearConstraint(fun=f, lb=-np.inf, ub=0)
            lhs.append(constraint)
        return lhs

    def forward(self, x):
        constraints = self.make_constraints(x=x)
        res = minimize(fun=self.objective,
                       x0=np.random.rand(self.n_objective + 1, ),
                       constraints=constraints)
        return res.x[:-1], res.x[-1]

    def __call__(self, x):
        return self.forward(x)


class LineSearch:
    def __init__(self,
                 objective_functions: List[Callable],
                 divider: float = 2,
                 step_size: float = 1,
                 sigma: float = 5e-1) -> None:
        self.divider = divider
        self.step_size = step_size
        self.objective_functions = objective_functions
        self.sigma = sigma

    def is_satisfy(self, x: Iterable, direction: Iterable, theta: float,
                   step_length: float) -> bool:
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


class Newton:
    def __init__(self,
                 tol: float = 1e-7,
                 max_iteration: int = 1000,
                 objectives: List[Callable] = []) -> None:
        self.tol = tol
        self.max_iteration = max_iteration
        self.objectives = objectives

    def add_problem(self, func: Callable):
        self.objectives.append(func)

    def init(self):
        self.line_search = LineSearch(objective_functions=self.objectives)
        self.direction_finder = NewtonDirection(func_list=self.objectives)

    def calc(self, x):
        res = []
        for f in self.objectives:
            res.append(f(x))
        return res

    def fit(self, x0: Iterable):
        assert len(self.objectives) > 0
        self.init()
        x = x0
        for idx in tqdm(range(self.max_iteration), leave=False):
            s, t = self.direction_finder(x)
            if t == 0:
                break
            step = self.line_search(x, s, t)
            x = x + step * s
        y = self.calc(x)
        return ResultReporter(x_min=x, y_min=y, n_iter=idx + 1)
