# -*- coding: utf-8 -*-
"""Householder's (extended Newton's) method
https://en.wikipedia.org/wiki/Householder%27s_method

Halley's method (d=2, fastest for the polynomials)
https://en.wikipedia.org/wiki/Halley%27s_method

Newton's method (d=1, stable, best for the arbitrary functions)
https://en.wikipedia.org/wiki/Newton%27s_method

See https://en.wikipedia.org/wiki/Householder%27s_method#Method for details
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from utils import tol


def roots(p: Poly, r: np.array, d: int = 1, rtol: float = 0, atol: float = 0) -> np.array:
    r = np.array(r)
    pd1 = i = 1
    pd = -p.deriv()
    while i < d:
        i += 1
        pd1 = pd
        pd = pd.deriv() * p - i * pd * p.deriv()
    pd1 = d * p * pd1
    dr = tol(np.abs(r), rtol, atol)
    while np.any(np.abs(dr) >= tol(np.abs(r), rtol, atol)):
        dr = pd1(r) / pd(r)
        r = r + dr
    return r
