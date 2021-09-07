# -*- coding: utf-8 -*-
"""Dandelin–Lobachesky–Graeffe method.
https://en.wikipedia.org/wiki/Graeffe%27s_method

Bounds on all roots
https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#Bounds_on_all_roots
"""

from . import *


def _maxn(a: np.array, n: int = 1) -> np.array:
    return a[np.argpartition(-a, n)[:n]]


def _abs_norm(a: np.array) -> np.array:
    return np.abs(a[:-1] / a[-1])


def _abs_norm_dim(a: np.array) -> np.array:
    b = _abs_norm(a)
    return b ** (1 / np.arange(len(b), 0, -1))


def _abs_ratio(a: np.array) -> np.array:
    a = a[a != 0]
    return np.abs(a[:-1] / a[1:])


_root_limit_formula = {
    "lagrange_weak": (lambda b: max(1, np.sum(b)), _abs_norm),
    "cauchy": (lambda b: 1 + np.max(b), _abs_norm),
    "zassenhaus": (lambda d: 2 * np.max(d), _abs_norm_dim),
    # originally given by Lagrange, but attributed to Zassenhaus by Donald Knuth
    "lagrange": (lambda d: np.sum(_maxn(d, 2)), _abs_norm_dim),
    # Lagrange improved previous with formula using sum(max(_,2)) instead of 2*max(_)
    "lagrange_ratio": (lambda c: np.sum(c), _abs_ratio),
    "hoelder": (lambda b: np.sqrt(1 + np.sum(b ** 2)), _abs_norm),
    "fujiwara": (lambda d: 2 * max(d[0] * 2 ** (-1 / (len(d))), np.max(d[1:])), _abs_norm_dim),
    "kojima": (lambda c: 2 * max(c[0] / 2, np.max(c[1:])), _abs_ratio)
    }


def root_limit(p: Poly, method=None):
    if method:
        f = _root_limit_formula[method]
        lim = f[0](f[1](p.coef))
    else:
        lim = np.inf
        for m in _root_limit_formula:
            f = _root_limit_formula[m]
            lim = min(lim, f[0](f[1](p.coef)))
    return lim


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


def estimate_roots(p: Poly):
    i = 0
    r = -p.coef[:-1] / p.coef[1:]
    while np.any(r < 0):
        p = dgiteration(p)
        i += 1
        r = -p.coef[:-1] / p.coef[1:]
    return r**(2**(-i))


def roots_classical(p: Poly, d: int):
    # TODO: process with complex roots
    q = [p]
    for i in range(d):
        q.append(dgiteration(q[i]))
    r = -q[-1].coef[:-1] / q[-1].coef[1:]
    # if __debug__:
    #     utils.show_roots(r, q[-1])
    # print(r, '\n')
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
    r = -p.coef[:-1] / p.coef[1:]
    re = -pe.coef[:-1] / pe.coef[1:]
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
