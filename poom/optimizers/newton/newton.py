from typing import Callable, Iterable, List

import torch
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from tqdm.auto import tqdm

from .utils import get_approx


class NewtonDirection:
    def __init__(self, func_list: List[Callable], n_objectives: int,
                 n_variables: int) -> None:
        self.objective = lambda x: x[-1]
        self.n_objective = n_objectives
        self.n_variables = n_variables
        self.func_list = func_list

    def __make_constraint(self, func: Callable, x):
        g, h = get_approx(func, x)

        def constraints(s):
            return np.transpose(g).dot(
                s[:-1]) + (np.transpose(s[:-1]).dot(h).dot(s[:-1])) / 2 - s[-1]

        return constraints

    def _get_theta(self, x: Iterable, s: Iterable) -> float:
        res = []
        for f in self.func_list:
            g, h = get_approx(f, x)
            theta = np.transpose(g).dot(
                s) + (np.transpose(s).dot(h).dot(s)) / 2
            res.append(theta)
        return max(res)

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
                       x0=np.random.rand(self.n_variables + 1, ),
                       constraints=constraints)
        s = res.x[:-1]
        theta = self._get_theta(x, s)
        return s, theta

    def __call__(self, x):
        return self.forward(x)


class LineSearch:
    def __init__(self,
                 objective_functions: List[Callable],
                 constraints: Callable = None,
                 divider: float = 2,
                 step_size: float = 1,
                 sigma: float = 5e-1) -> None:
        self.divider = divider
        self.step_size = step_size
        self.constraints = constraints
        self.objective_functions = objective_functions
        self.sigma = sigma

    def __is_inbound(self, x: Iterable) -> bool:
        if self.constraints is None:
            return True
        status = self.constraints(x)
        return status

    def __is_significant_value(self, x_new: Iterable, x_old: Iterable,
                               step_length: float, theta: float):
        for f in self.objective_functions:
            ref_values = f(x_new) - f(x_old) - self.sigma * step_length * theta
            if ref_values > 0:
                return False
        return True

    def is_satisfy(self, x: Iterable, direction: Iterable, theta: float,
                   step_length: float) -> bool:
        x_new = x + step_length * direction
        is_significant_value = self.__is_significant_value(
            x_new=x_new, x_old=x, step_length=step_length, theta=theta)
        is_satisfy_constraitns = self.__is_inbound(x_new)
        return is_satisfy_constraitns and is_significant_value

    def armijo(self, x: Iterable, direction: Iterable, theta: float):
        step_length = self.step_size
        for _ in range(100):
            if self.is_satisfy(
                    x=x, direction=direction, theta=theta,
                    step_length=step_length) is True:
                return step_length
            step_length /= 2
        return 0

    def __call__(self, x, direction, theta):
        return self.armijo(x, direction, theta)


class Newton:
    def __init__(self,
                 tol: float = 1e-12,
                 max_iteration: int = 1000,
                 n_variables: int = 3,
                 n_objectives: int = 2,
                 objectives: List[Callable] = [],
                 constraints: Callable = None) -> None:
        self.tol = tol
        self.n_objective = n_objectives
        self.n_variables = n_variables
        self.max_iteration = max_iteration
        self.objectives = objectives
        self.constraints = constraints

    def add_problem(self, func: Callable):
        self.objectives.append(func)

    def init(self):
        self.line_search = LineSearch(objective_functions=self.objectives,
                                      constraints=self.constraints)
        self.direction_finder = NewtonDirection(func_list=self.objectives,
                                                n_objectives=self.n_objective,
                                                n_variables=self.n_variables)

    def calc(self, x):
        res = []
        for f in self.objectives:
            res.append(f(x))
        return res

    def fit(self, x0: Iterable):
        assert len(self.objectives) > 0
        self.init()
        x = x0
        res = []
        y = self.calc(x)
        res.append(y)
        for _ in tqdm(range(self.max_iteration), leave=False):
            s, t = self.direction_finder(x)
            step = self.line_search(x, s, t)
            if t == 0:
                break
            x = x + step * s
            y = self.calc(x)
            res.append(y)
        return res, x

    def find_pareto_front(self, x: List[Iterable]):
        output = []
        argmin = []
        for x0 in tqdm(x):
            reporter, _x = self.fit(x0)
            output.append(reporter)
            argmin.append(_x)
        return output, np.array(argmin)
