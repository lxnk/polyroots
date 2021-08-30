# -*- coding: utf-8 -*-
"""Dandelin–Lobachesky–Graeffe method.
https://en.wikipedia.org/wiki/Graeffe%27s_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
import utils

def dgiteration(p: Poly):
    """Dandelin–Graeffe iteration
    Polynomial q(x) has root, which are the squares of the roots of polynomial p(x)

    .. math::
        p(x)   = \\sum_{k=0} ^n a_k x^k
    .. math::
        q(x^2)   = p(-x)p(x)
    .. math::
        q(x)   = \\sum_{k=0} ^n b_k x^k
    .. math::
        b_k = 2 (-1)^k a_k^2 + \\sum_{i=1} ^{min(k,n-k)} (-1)^{k-i}a_{k-i}a_{k+i}

    New roots are at the larger distance
    """
    n = p.degree()
    n2 = int((n+1)/2)
    m = np.ones_like(p.coef)
    m[1::2] = -1
    q = p.copy()

    for k in range(-n2, n + 1 - n2):
        if k >= 0:
            q.coef[k] = np.sum(p.coef[:2*k+1] * p.coef[2*k::-1]
                               * m[2*k::-1])
        else:
            q.coef[k] = np.sum(p.coef[2*k+1:] * p.coef[:2*k:-1]
                               * m[:2*k:-1])
    return q


def roots_classical(p: Poly, d: int):
    # TODO: process with complex roots
    q = [p]
    for i in range(d):
        q.append(dgiteration(q[i]))
    r = -q[-1].coef[:-1] / q[-1].coef[1:]
    # if __debug__:
    #     utils.show_roots(r, q[-1])
    # print(r,'\n')
    for i in range(d-1, -1, -1):
        r = np.lib.scimath.sqrt(r)
        c = np.abs(q[i](-r)) < np.abs(q[i](r))
        # print(r)
        # print(np.abs(q[i](r)))
        # print(np.abs(q[i](-r)), '\n')
        r[c] *= -1
        # if __debug__:
        #     utils.show_roots(r, q[i])
    return r


def roots_tangential(p: Poly, d: int, eps: float = 1e-5):
    # TODO: process with complex roots
    pe = p+eps * p.deriv()
    for i in range(d):
        p = dgiteration(p)
        pe = dgiteration(pe)
    r = -p.coef[:-1]/p.coef[1:]
    re = -pe.coef[:-1]/pe.coef[1:]
    r = -2**d*eps*r/(re-r)
    return r
