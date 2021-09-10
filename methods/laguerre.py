# -*- coding: utf-8 -*-
"""Laguerre's method.
https://en.wikipedia.org/wiki/Laguerre%27s_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from utils import tol


def roots(p: Poly, r: np.array, rtol: float = 0, atol: float = 0) -> np.array:
    r = np.array(r)
    pd = p.deriv()
    pdd = pd.deriv()
    n = p.degree()
    dr = tol(np.abs(r), rtol, atol)
    while np.any(np.abs(dr) >= tol(np.abs(r), rtol, atol)):
        f = p(r)
        g = pd(r)
        # print("f=", f, "g=", g)
        # print("den=", (n-1)*((n-1)*g**2 - n*pdd(r)*f))
        s = np.lib.scimath.sqrt((n-1)*((n-1)*g**2 - n*pdd(r)*f))
        # print("s=", s)
        dd = np.abs(g+s) < np.abs(g-s)
        dr = - n * f / (g + (-1)**dd * s)
        r = r + dr
    return r
