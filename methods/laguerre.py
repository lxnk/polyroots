# -*- coding: utf-8 -*-
"""Laguerre's method.
https://en.wikipedia.org/wiki/Laguerre%27s_method
"""

from . import *


def roots(p: Poly, r: np.array, rtol: float = 0, atol: float = 0) -> np.array:
    r = np.array(r)
    pd = p.deriv()
    pdd = pd.deriv()
    n = p.degree()
    dr = tol(np.abs(r), rtol, atol)
    while np.any(np.abs(dr) >= tol(np.abs(r), rtol, atol)):
        f = p(r)
        g = pd(r)
        h = g**2 - pdd(r) / p(r)
        s = np.sqrt((n-1)*((n-1)*g**2 - n*pdd(r)*f))
        dd = np.abs(g+s) < np.abs(g-s)
        dr = - n * f / (g + (-1)**dd * s)
        r = r + dr
    return r
