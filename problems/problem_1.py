from typing import Iterable

import numpy as np


def obj_1(x: Iterable):
    return x[0] + x[1]**2 + (x[0]**2 + x[1]**2)*2


def obj_2(x: Iterable):
    return x[0] - x[1]**2 + (x[0]**2 + x[1])


objs = [obj_1, obj_2]
