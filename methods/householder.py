# -*- coding: utf-8 -*-
"""Householder's (extended Newton's) method
https://en.wikipedia.org/wiki/Householder%27s_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np

eps = np.finfo(float).tiny


def roots(p: Poly, x: float, d: int = 1, abstol: float = eps, reltol: float = 0) -> Poly:
    pd1 = i = 1
    pd = -p.deriv()
    while i < d:
        i += 1
        pd1 = pd
        pd = pd.deriv() * p - i * pd * p.deriv()
    pd1 = d * p * pd1
    tol = lambda t: np.maximum(abstol + reltol * t, np.spacing(t))
    dx = tol(np.abs(x))
    while np.abs(dx) >= tol(np.abs(x)):
        dx = pd1(x) / pd(x)
        x = x + dx
    return x, itr
