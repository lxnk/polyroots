# -*- coding: utf-8 -*-
"""Find all real positive roots
"""
from numpy.polynomial import Polynomial as Poly
from methods import graeffe, householder, laguerre, bounds
import numpy as np
from utils import unique as unique_in_sort


def roots_numpy(p: Poly) -> np.array:
    r = p.roots()
    return r[(r.imag == 0) & (r.real >= 0)].real


def roots_graeffe_householder(p: Poly, d: int, rtol: float = 0, atol: float = 0) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = householder.roots(p, r, d=d, rtol=rtol, atol=atol)
    r = np.unique(r[r >= 0])
    return r


def roots_graeffe_newton(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    return roots_graeffe_householder(p, 1, rtol=rtol, atol=atol)


def roots_graeffe_halley(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    return roots_graeffe_householder(p, 2, rtol=rtol, atol=atol)


def roots_graeffe_laguerre(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = laguerre.roots(p, r, rtol=rtol, atol=atol)
    r = unique_in_sort(np.sort(r[(r.imag == 0) & (r.real >= 0)].real), rtol=rtol, atol=atol)
    return r


def roots_graeffe_lim_laguerre(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    rmax = bounds.root_limit(p, method="lagrange", rproots=True)
    # print("rmax  =", rmax)
    r = np.append(r[r < rmax], rmax)
    r = laguerre.roots(p, r, rtol=rtol, atol=atol)
    r = unique_in_sort(np.sort(r[(r.imag == 0) & (r.real >= 0)].real), rtol=rtol, atol=atol)
    return r