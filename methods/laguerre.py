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
        g = pd(r) / p(r)
        h = g**2 - pdd(r) / p(r)
        dr = - n /(g + np.sqrt((n-1)*(n*h - g**2))*np.sign(g))
        r = r + dr
    return r
