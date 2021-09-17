# -*- coding: utf-8 -*-
"""Find all real positive roots
"""
from numpy.polynomial import Polynomial as Poly
from methods import graeffe, householder, laguerre, bounds, vincent
import numpy as np
from utils import unique as unique_in_sort
from more_itertools import pairwise


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
    r = np.append(r[r < rmax], rmax)
    r = laguerre.roots(p, r, rtol=rtol, atol=atol)
    r = unique_in_sort(np.sort(r[(r.imag == 0) & (r.real >= 0)].real), rtol=rtol, atol=atol)
    return r


def roots_graeffe_lim_vincent(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = np.sort(r)
    rmax = bounds.root_limit(p, method="lagrange", rproots=True)
    r = np.append(r[r < rmax], rmax)
    sr = np.insert(np.append(np.sqrt(r[:-1] * r[1:]), rmax), 0, 0)
    iv = list()
    for it in pairwise(sr):
        iv.append(it)
    r = vincent.root_intervals_bisection(p, iv=iv)
    r.sort()
    return r


def roots_graeffe_lim_vincent_newton(p: Poly, rtol: float = 0, atol: float = 0) -> np.array:
    r = graeffe.roots_classical(p, d=0, absval=True)
    r = np.sort(r)
    rmax = bounds.root_limit(p, method="lagrange", rproots=True)
    # print("rmax  =", rmax)
    r = np.append(r[r < rmax], rmax)
    sr = np.insert(np.append(np.sqrt(r[:-1] * r[1:]), rmax), 0, 0)
    iv = list()
    for it in pairwise(sr):
        iv.append(it)
    ri = vincent.root_intervals_bisection(p, iv=iv)
    ri.sort()
    print(ri)
    # r0 = list(map(lambda i: np.sqrt(np.prod(i)), ri))
    # r0 = list(map(lambda i: i[1], ri))
    r0 = list(map(lambda i: np.sum(i) / 2, ri))

    print(r0)
    # r = householder.roots(p, r0, d=2, rtol=rtol, atol=atol)
    # r.sort()
    return r
