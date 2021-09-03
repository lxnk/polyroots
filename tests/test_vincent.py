# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
from methods import vincent


@pytest.fixture(params=[(-3, 4, -4, 2)], ids=["c-+-+"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-.3, .49, .52, .21)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


def test_sign_var_num(polyc):
    n = vincent.sign_var_num(polyc)
    print(n)
    nt.assert_equal(n, 3)


def test_root_intervals_cfrac(polyr):
    iv = vincent.root_intervals_cfrac(polyr)
    print('\n', iv, '<->', polyr.roots())


def test_root_intervals_bisection(polyr):
    iv = vincent.root_intervals_bisection(polyr)
    print('\n', iv, '<->', polyr.roots())

