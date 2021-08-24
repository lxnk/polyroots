# -*- coding: utf-8 -*-
"""Aberth-Ehrlich method.
https://en.wikipedia.org/wiki/Aberth_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from utils import tol


def roots(p: Poly, rtol: float = 0, atol: float = 0):
    pd = p.deriv()
    r = (0.8 + 0.6j) ** np.r_[1:p.degree()+1]
    dr = tol(np.abs(r), rtol, atol) + 0j
    while np.any(np.abs(dr) >= tol(np.abs(r), rtol, atol)):
        for i, z in enumerate(r):
            dr[i] = - (p(z) / pd(z))
            dr[i] = dr[i] / (1 + dr[i] * np.sum(1.0/(z - np.delete(r, i))))
        r = r + dr
    return r
