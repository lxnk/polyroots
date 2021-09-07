# -*- coding: utf-8 -*-
"""Dandelin–Lobachesky–Graeffe method.
https://en.wikipedia.org/wiki/Graeffe%27s_method

Bounds on all roots
https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#Bounds_on_all_roots
"""

from . import *


def _maxn(a: np.array, n: int = 1) -> np.array:
    return a[np.argpartition(-a, n)[:n]]


def _norm(a: np.array) -> np.array:
    return a[:-1] / a[-1]


def _abs(a: np.array) -> np.array:
    return np.abs(a)


def _dim(a: np.array) -> np.array:
    return a ** (1 / np.arange(len(a), 0, -1))


def _ratio(a: np.array) -> np.array:
    a = a[a != 0]
    if len(a) == 1:
        return np.array([0])
    else:
        return a[:-1] / a[1:]


def _half(a: np.array) -> np.array:
    a[0] /= 2
    return a


_root_limit_formula = {
    "lagrange_weak": (_norm, _abs, lambda b: max(1, np.sum(b))),
    "cauchy": (_norm, _abs, lambda b: 1 + np.max(b)),
    "zassenhaus": (_norm, _abs, _dim, lambda d: 2 * np.max(d)),
    # originally given by Lagrange, but attributed to Zassenhaus by Donald Knuth
    "lagrange": (_norm, _abs, _dim, lambda d: np.sum(_maxn(d, 2))),
    # Lagrange improved previous with formula using sum(max(_,2)) instead of 2*max(_)
    "lagrange_ratio": (_ratio, _abs, lambda c: np.sum(c)),
    "hoelder": (_norm, _abs, lambda b: np.sqrt(1 + np.sum(b ** 2))),
    "fujiwara": (_norm, _half, _abs, _dim, lambda d: 2 * np.max(d)),
    "kojima": (_ratio, _half, _abs, lambda c: 2 * np.max(c))
    }


def root_limit(p: Poly, method, rproots = False):
    a = p.coef.copy()
    if rproots:
        if a[-1] < 0:
            a *= -1
        an = a[-1]
        a[a > 0] = 0
        a[-1] = an

    for f in _root_limit_formula[method]:
        a = f(a)
    return a


# TODO: Estimate multiplicity


def dgiteration(p: Poly):
    """Dandelin–Graeffe iteration
    Polynomial `q(x)` has root, which are the squares of the roots of polynomial `p(x)`

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
    n2 = int((n + 1) / 2)
    m = np.ones_like(p.coef)
    m[1::2] = -1
    q = p.copy()

    for k in range(-n2, n + 1 - n2):
        if k >= 0:
            q.coef[k] = np.sum(p.coef[:2 * k + 1] * p.coef[2 * k::-1]
                               * m[2 * k::-1])
        else:
            q.coef[k] = np.sum(p.coef[2 * k + 1:] * p.coef[:2 * k:-1]
                               * m[:2 * k:-1])
    return q


def roots_classical(p: Poly, d: int = 0, absval = False):
    # TODO: process with complex roots
    q = [p]
    for i in range(d):
        q.append(dgiteration(q[i]))
    r = -_ratio(q[-1].coef)
    # if __debug__:
    #     utils.show_roots(r, q[-1])
    # print(r, '\n')
    if absval:
        return np.abs(r) ** (2 ** (-d))
    else:
        for i in range(d - 1, -1, -1):
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
    pe = p + eps * p.deriv()
    for i in range(d):
        p = dgiteration(p)
        pe = dgiteration(pe)
    r = -_ratio(p.coef)
    re = -_ratio(pe.coef[:-1])
    r = -2 ** d * eps * r / (re - r)
    return r


def roots_tangential(p: Poly, d: int, eps: float = 1e-5):
    # TODO: process with complex roots
    pe = p + eps * p.deriv()
    for i in range(d):
        p = dgiteration(p)
        pe = dgiteration(pe)
    r = -p.coef[:-1] / p.coef[1:]
    re = -pe.coef[:-1] / pe.coef[1:]
    r = -2 ** d * eps * r / (re - r)
    return r
