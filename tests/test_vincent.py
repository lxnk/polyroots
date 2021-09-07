# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
from methods import vincent


@pytest.fixture
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-.6, -.3, .49, .52, .21)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


@pytest.mark.parametrize("polyc, num", [((-3, 4, -4, 2), 3)], indirect=["polyc"], ids=["c-+-+"])
def test_sign_var_num(polyc, num):
    n = vincent.sign_var_num(polyc)
    nt.assert_equal(n, num)


def test_root_intervals_cfrac(polyr):
    """Last data
    [(-1, -1/2), (-1/2, 0.0), (1/2, 1), (1/3, 1/2), (0.0, 1/3)]
    """
    iv = vincent.root_intervals_cfrac(polyr)
    r = polyr.roots()
    # print('\n', iv, '<->', r)
    for a, b in iv:
        nt.assert_equal(len(r[(r >= a) & (r <= b)]), 1)


def test_root_intervals_bisection(polyr):
    """Last data
    [(1/2, 1), (1/4, 1/2), (0,1/4)]
    """
    iv = vincent.root_intervals_bisection(polyr)
    r = polyr.roots()
    # print('\n', iv, '<->', polyr.roots())
    for a, b in iv:
        nt.assert_equal(len(r[(r >= a) & (r <= b)]), 1)

