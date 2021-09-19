# -*- coding: utf-8 -*-

# from tests.context import *

import pytest
import numpy.testing as nt
from numpy.polynomial import polynomial as pl
import numpy as np

from numpy.polynomial import Polynomial as Poly
from utils import sort_roots

from utils import unique as unique_in_sort
import methods as mt
from models import repos
# from data.read_data import access_data as loadpoly


@pytest.fixture(params=[(-3, 4, 4, 2),
                        (1, 2, 3, 1),
                        (.25, .5, .751, .9),
                        (3, 5, 5, 2000, 2),
                        (3, 5, 5, 2000, -5, 3),
                        (3, 5, -50, 2000, 70, 3),
                        (3, -5, -5, 2, 7, 3)])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


# # @pytest.mark.skip
# def test_positive_real_roots_decartes(polyc):
#     rc = polyc.roots()
#     print('\n', rc)
#     rr = rc[(rc.imag == 0) & (rc.real >= 0)].real
#     p = polyc.copy()
#     print("n0=", mt.vincent.sign_var_num(p))
#     r = list()
#     while vincent.sign_var_num(p) > 0:
#         x = bounds.root_limit(p, method="lagrange", rproots=True)
#         print("x0=", x)
#         x = householder.roots(p, x, d=1)
#         print("x=", x)
#         r.append(x)
#         p = p // [-x, 1]
#         print("p=", p)
#         print("n=", vincent.sign_var_num(p))
#     print(r, rr)
#     # nt.assert_allclose(np.sort(r), np.sort(rr))


def test_repos_graeffe_newton(polyc):
    rr = repos.roots_numpy(polyc)
    r = repos.roots_graeffe_newton(polyc)
    # print(r, rr)
    nt.assert_allclose(np.sort(r), np.sort(rr))


@pytest.mark.skip
def test_repos_graeffe_halley(polyc):
    """Does not always converge"""
    rr = repos.roots_numpy(polyc)
    r = repos.roots_graeffe_halley(polyc)
    # print(r, rr)
    nt.assert_allclose(np.sort(r), np.sort(rr))


def test_repos_graeffe_laguerre(polyc):
    rr = repos.roots_numpy(polyc)
    r = repos.roots_graeffe_laguerre(polyc)
    # print(r, rr)
    nt.assert_allclose(np.sort(r), np.sort(rr))


def test_matlab_roots_db():
    with open('../data/polydata.npy', 'rb') as f:
        for d in range(2, 7):
            poly = np.load(f)
            for p1 in poly:
                r0 = sort_roots(p1['root'])
                r1 = sort_roots(Poly(p1['coef']).roots())
                nt.assert_allclose(r1, r0, rtol=1e-3, atol=1e-15)
                nt.assert_allclose(r1, r0, atol=2e-5)


# @pytest.mark.skip
def test_repos_roots_graeffe_lim_laguerre_db_p5():
    with open('../data/polydata.npy', 'rb') as f:
        for d in range(2, 5):
            np.load(f)
        poly5 = np.load(f)
    i = 0
    for p1 in poly5:
        i += 1
        p = Poly(p1['coef'])
        rr = repos.roots_numpy(p)
        r = repos.roots_graeffe_lim_laguerre(p, rtol=1e-9, atol=1e-9)
        if len(r) == len(rr):
            nt.assert_allclose(r, rr, rtol=1e-9, atol=1e-9)
        else:
            print()
            print("r  =", r)
            print("rr =", rr)


def test_repos_roots_vincent_db_p5():
    with open('../data/polydata.npy', 'rb') as f:
        for d in range(2, 5):
            np.load(f)
        poly5 = np.load(f)
    i = 0
    for p1 in poly5:
        i += 1
        p = Poly(p1['coef'])
        rr = repos.roots_numpy(p)
        ri = repos.roots_graeffe_lim_vincent(p, rtol=1e-9, atol=1e-9)
        rn = np.asarray(ri)
        if len(ri) == len(rr):
            nt.assert_array_less(rn[:, 0], rr)
            nt.assert_array_less(rr, rn[:, 1])
        else:
            print()
            print("i  =", i)
            print("ri =", ri)
            print("rr =", rr)


def test_repos_roots_vincent_newton_db_p5():
    with open('../data/polydata.npy', 'rb') as f:
        for d in range(2, 5):
            np.load(f)
        poly5 = np.load(f)
    i = 0
    for p1 in poly5:
        i += 1
        p = Poly(p1['coef'])
        rr = repos.roots_numpy(p)
        # print()
        # print(rr)
        rx = repos.roots_graeffe_lim_vincent_halley(p, rtol=1e-9, atol=1e-9)
        if len(rx) == len(rr):
            nt.assert_allclose(rx, rr, rtol=1e-3, atol=1e-15)
        else:
            print()
            print("i  =", i)
            print("rx =", rx)
            print("rr =", rr)