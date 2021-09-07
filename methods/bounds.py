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

