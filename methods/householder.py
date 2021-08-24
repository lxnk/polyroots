# -*- coding: utf-8 -*-
"""Householder's (extended Newton's) method
https://en.wikipedia.org/wiki/Householder%27s_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from utils import tol


def roots(p: Poly, r: float, d: int = 1, rtol: float = 0, atol: float = 0) -> np.array:
    pd1 = i = 1
    pd = -p.deriv()
    while i < d:
        i += 1
        pd1 = pd
        pd = pd.deriv() * p - i * pd * p.deriv()
    pd1 = d * p * pd1
    dr = tol(np.abs(r), rtol, atol)
    while np.abs(dr) >= tol(np.abs(r), rtol, atol):
        dr = pd1(r) / pd(r)
        r = r + dr
    return r
