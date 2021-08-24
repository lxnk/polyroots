# -*- coding: utf-8 -*-
"""Durand–Kerner (Weierstrass) method.
https://en.wikipedia.org/wiki/Durand–Kerner_method
"""


from numpy.polynomial import Polynomial as Poly
import numpy as np


def roots(p: Poly, rtol: float = 0, atol: float = 0):
    def tol(t):
        return np.maximum(atol + rtol * t, np.spacing(t))

    p = p / p.coef[p.degree()]
    r = (0.8 + 0.6j) ** np.r_[1:p.degree() + 1]
    dr = tol(np.abs(r)) + 0j
    while np.any(np.abs(dr) >= tol(np.abs(r))):
        for i, z in enumerate(r):
            dr[i] = -p(z) / np.prod(z - np.delete(r, i))
        r = r + dr
    return np.sort(r)

