# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
import numpy as np
from methods import bounds
from methods.graeffe import roots_classical as estimate_roots


@pytest.fixture(params=[(-3, 4, -4, 2)], ids=["c-3442"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-3, 4, 5, 2)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


@pytest.mark.parametrize("a,b,c,c2,d,d2", [([3, -7, 0, -1, 5],
                                      [.6, 1.4, 0, .2],
                                      [3 / 7, 7, 1 / 5],
                                      [3 / 14, 7, 1 / 5],
                                      [.6 ** (1 / 4), 1.4 ** (1 / 3), 0, .2],
                                      [.3 ** (1 / 4), 1.4 ** (1 / 3), 0, .2]),
                                     ([1, 0, 4, -3, 0, 5, -2],
                                      [.5, 0, 2, 1.5, 0, 2.5],
                                      [1/4, 4 / 3, 3 / 5, 5 / 2],
                                      [1/8, 4 / 3, 3 / 5, 5 / 2],
                                      [.5 ** (1 / 6), 0, 2 ** (1 / 4), 1.5 ** (1 / 3), 0, 2.5],
                                      [.25 ** (1 / 6), 0, 2 ** (1 / 4), 1.5 ** (1 / 3), 0, 2.5])], ids=["a5", "a7"])
def test__abs_x(a, b, c, c2, d, d2):
    nt.assert_array_equal(bounds._abs(bounds._norm(np.array(a))), b)
    nt.assert_array_equal(bounds._abs(bounds._ratio(np.array(a))), c)
    nt.assert_array_equal(bounds._abs(bounds._half(bounds._ratio(np.array(a)))), c2)
    nt.assert_array_equal(bounds._dim(bounds._abs(bounds._norm(np.array(a)))), d)
    nt.assert_array_equal(bounds._dim(bounds._abs(bounds._half(bounds._norm(np.array(a))))), d2)


def test_root_limit(polyr):
    maxabsr = np.max(np.abs(polyr.roots()))
    for m in bounds._root_limit_formula:
        rlim = bounds.root_limit(polyr, method=m)
        nt.assert_array_less(maxabsr, rlim)


@pytest.mark.parametrize("polyc", [(3, 5, 5, 2000, 2),
                                   (3, 5, 5, 2000, -5, 3),
                                   (3, 5, -50, 2000, 70, 3)], indirect=["polyc"], ids=["ml1", "ml2", "ml3"])
def test_real_root_limit(polyc):
    re = np.sort(estimate_roots(polyc, 4, absval=True))
    rr = polyc.roots()
    r0 = np.max(np.sort(np.abs(rr)))
    rr = np.max(np.sort(rr[rr.imag == 0].real))
    # print(r0, rr)
    # print(re, graeffe.root_limit(polyc, method="lagrange"))
    nt.assert_array_less(re, bounds.root_limit(polyc, method="lagrange"))

    for m in bounds._root_limit_formula:
        rlim0 = bounds.root_limit(polyc, method=m)
        rlimr = bounds.root_limit(polyc, method=m, rproots=True)
        # print(m, '\t', rlim0, '\t', rlimr)
        nt.assert_array_less(r0, rlim0)
        nt.assert_array_less(rr, rlimr)


def test_estimate_roots(polyr):
    re= np.sort(estimate_roots(polyr, 4, absval=True))
    r0 = np.sort(np.abs(polyr.roots()))
    # print()
    # print(r0)
    # print(re, graeffe.root_limit(polyr, method="lagrange"))
    nt.assert_allclose(re, r0, rtol=.1)
    nt.assert_array_less(re, bounds.root_limit(polyr, method="lagrange"))
