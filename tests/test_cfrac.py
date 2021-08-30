import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial as pl
from methods import cfrac


@pytest.fixture(params=[(-3, 4, 4, 2)], ids=["c-++-"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-3, 4.9, 5.2, 2.1)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


def test_sign_var_num(polyc):
    n = cfrac.sign_var_num(polyc)
    print(n)
    nt.assert_equal(n, 3)


def test_root_intervals(polyr):
    iv = cfrac.root_intervals(polyr)
    print('\n', iv, '<->', polyr.roots())
