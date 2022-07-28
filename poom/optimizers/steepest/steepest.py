import numpy as np
from typing import Any, Callable
from scipy.optimize import fmin
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class Obj:
    def f(self, x):
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x):
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x):
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])


# contrained conditions


@dataclass
class SteepestDescent:
    ndim: int
    nu: float  #alpha
    sigma: float  #m1
    eps: float

    def grad(self, f, x, h=1e-4):
        g = np.zeros_like(x)
        for i in range(self.ndim):
            tmp = x[i]
            x[i] = tmp + h
            yr = f(x)
            x[i] = tmp - h
            yl = f(x)
            g[i] = (yr - yl) / (2 * h)
            x[i] = tmp
        return g

    def nabla_F(self, x):
        obj = Obj()
        F = obj.Fss()
        nabla_F = np.zeros((len(F), self.ndim))  # (m, n) dimensional matrix
        for i, f in enumerate(F):
            nabla_F[i] = self.grad(F[i], x)
        return nabla_F

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(np.dot(nabla_F, d)) + 0.5 * np.linalg.norm(d)**2

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * np.linalg.norm(d)**2

    def armijo(self, d, x):
        obj = Obj()
        t = 1
        Fl = np.array(obj.Fs(x + t * d))
        Fr = np.array(obj.Fs(x))
        Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        while np.all(Fl > Fr + Re):
            t *= self.nu
            Fl = np.array(obj.Fs(x + t * d))
            Fr = np.array(obj.Fs(x))
            Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        return t

    def steepest(self, x):
        obj = Obj()
        list_point = []
        d = np.array(fmin(self.phi, x, args=(x, )))
        th = self.theta(d, x)
        for i in range(1000):
            th = self.theta(d, x)
            t = self.armijo(d, x)
            y = obj.Fs(x)
            list_point.append(y)
            x = x + t * d
            d = np.array(fmin(self.phi, x, args=(x, )))
            if abs(th) > self.eps:
                break
        return list_point