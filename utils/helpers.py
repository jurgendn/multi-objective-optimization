import numpy as np


def make_mesh(low: float = -2,
              high: float = 2,
              n_points: int = 51,
              n_variables: int = 2):
    r = np.linspace(start=low, stop=high, num=n_points).reshape(1, -1)
    axes = np.full(shape=(n_variables, n_points), fill_value=r)
    out = np.meshgrid(*axes)
    out = np.array(out)
    out = out.reshape(n_variables, -1).T
    return out
