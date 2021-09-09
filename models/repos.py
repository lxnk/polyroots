# -*- coding: utf-8 -*-
"""Find all real positive roots
"""
from numpy.polynomial import Polynomial as Poly
from methods import graeffe, householder, laguerre
import numpy as np


def roots_numpy(p: Poly) -> np.array:
    r = p.roots()
    # print('\n', r)
    return r[(r.imag == 0) & (r.real >= 0)].real


def roots_graeffe_householder(p: Poly) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = householder.roots(p, r, d=1)
    r = np.unique(r[r >= 0])
    return r


def roots_graeffe_laguerre(p: Poly) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = laguerre.roots(p, r)
    r = np.unique(r[(r.imag == 0) & (r.real >= 0)].real)
    return r
