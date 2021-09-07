# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
import numpy as np
from methods import graeffe
from utils import sort_roots


@pytest.fixture(params=[(-3, 4, -4, 2)], ids=["c-3442"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-3, 4, 5, 2)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


def test_dgiteration(polyc):
    pp = polyc.copy()
    pp.coef[1::2] *= -1
    pp = pp * polyc
    pg = graeffe.dgiteration(polyc)
    nt.assert_array_equal(pp.coef[::2], pg.coef)


def test_roots_classical_real_roots(polyr):
    polyr.coef /= polyr.coef[polyr.degree()]
    r = graeffe.roots_classical(polyr, 3)
    # print('\n', sort_roots(r), '\n', sort_roots(polyr.roots()))
    nt.assert_allclose(sort_roots(r), sort_roots(polyr.roots()), rtol=3e-2)


def test_roots_tangential(polyr):
    polyr.coef /= polyr.coef[polyr.degree()]
    r = graeffe.roots_tangential(polyr, 5, 1e-3)
    # print('\n', sort_roots(r), sort_roots(polyr.roots()))
    nt.assert_allclose(sort_roots(r), sort_roots(polyr.roots()), rtol=3e-2)


@pytest.mark.parametrize("a,b,c,d", [([3, -7, 0, -1, 5],
                                      [.6, 1.4, 0, .2],
                                      [3 / 7, 7, 1 / 5],
                                      [.6 ** (1 / 4), 1.4 ** (1 / 3), 0, .2]),
                                     ([0, 4, -3, 0, 5, -2],
                                      [0, 2, 1.5, 0, 2.5],
                                      [4 / 3, 3 / 5, 5 / 2],
                                      [0, 2 ** (1 / 4), 1.5 ** (1 / 3), 0, 2.5])], ids=["a5", "a6"])
def test__abs_x(a, b, c, d):
    nt.assert_array_equal(graeffe._abs_norm(np.array(a)), b)
    nt.assert_array_equal(graeffe._abs_ratio(np.array(a)), c)
    nt.assert_array_equal(graeffe._abs_norm_dim(np.array(a)), d)


def test_root_limit(polyr):
    maxabsr = np.max(np.abs(polyr.roots()))
    for m in graeffe._root_limit_formula:
        rlim = graeffe.root_limit(polyr, method=m)
        nt.assert_array_less(maxabsr, rlim)


def test_estimate_roots(polyr):
    r = np.sort(graeffe.estimate_roots(polyr))
    r0 = np.sort(np.abs(polyr.roots()))
    print()
    print(r0)
    print(r, graeffe.root_limit(polyr, method="lagrange"))
    nt.assert_allclose(r, r0, rtol=.5)
    nt.assert_array_less(r, graeffe.root_limit(polyr, method="lagrange"))
