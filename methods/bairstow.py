# -*- coding: utf-8 -*-
"""
Bairstow's method
https://en.wikipedia.org/wiki/Bairstow%27s_method

In this article also explained now to quickly compute the quotient polynomial coefficients. We do not need it while
polynomials divmod is already implemented. The rest is 2D Newton method. For real roots one can use Newton method, and
to calculate the coefficients of the quotient polynomial one may use Horner's method
https://en.wikipedia.org/wiki/Horner%27s_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from utils import tol


def quadratic_root_divmod(p: Poly, r, rtol: float = 0, atol: float = 0) -> tuple:
    """
    Bairstow's method

    Values of `u` and `v` for which this occurs can be discovered by picking starting values and iterating Newton's
    method in two dimensions

    .. math::
        U = \\begin{pmatrix}
                gu-h & -g \\\\
                gv   & -h
            \\end{pmatrix}

    .. math::
        (u, v)^T = (u, v)^T - U^{-1} (u, v)^T

    """
    pd = Poly([abs(r)**2, -2*r.real, 1], domain=p.domain, window=p.window)
    md = np.zeros((2,2))
    duv = tol(np.abs(pd.coef[:-1]), rtol, atol)
    while np.any(np.abs(duv) >= tol(np.abs(pd.coef[:-1]), rtol, atol)):
        q, rd1 = divmod(p, pd)
        # print(rd1)
        rd2 = q % pd
        md[1, 1] = rd2.coef[1] * pd.coef[1] - rd2.coef[0]
        md[1, 0] = -rd2.coef[1]
        md[0, 1] = rd2.coef[1] * pd.coef[0]
        md[0, 0] = -rd2.coef[0]
        duv = -np.linalg.solve(md, rd1.coef if rd1.degree() == 1 else np.array([rd1.coef[0], 0]))
        pd.coef[:-1] += duv
    return (-pd.coef[1]/2 + np.sqrt(pd.coef[0] - pd.coef[1]**2 / 4) * 1j, q)
