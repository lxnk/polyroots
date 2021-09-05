# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
import numpy as np
from methods import bairstow

_root = [(1, -1+1j, -1-1j, 7, 3), (-2, -2+3j, -2-3j, 2, 7)]
_rini = [-1.1-.9j, -1.5-2.0j]
_ids = ["r-1-1173", "r-2-2327"]

fail_case = pytest.param((-2, -2+3j, -2-3j, 2, 7), -2-2.0j,
                         marks=pytest.mark.xfail(reason="Other root is met?"), id='jump')


@pytest.fixture
def polyr(request):
    """Create polynomial out of the real roots"""
    p = Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])
    p.coef = p.coef.real
    return p


# @pytest.mark.parametrize("polyr, rini", [*zip(_root, _rini), fail_case], indirect=["polyr"], ids=[*_ids, None])
@pytest.mark.parametrize("polyr, rini", zip(_root, _rini), indirect=["polyr"], ids=_ids)
def test_quadratic_divider(polyr, rini):
    r, _ = bairstow.quadratic_root_divmod(polyr, rini)
    r0 = polyr.roots()
    i = np.abs(r0-r).argmin()
    # print(r0[i], r)
    nt.assert_allclose(r0[i], r)


@pytest.mark.parametrize("rtol, atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
@pytest.mark.parametrize("polyr, rini", zip(_root, _rini), indirect=["polyr"], ids=_ids)
def test_roots_tol(polyr, rini, rtol, atol):
    r, _ = bairstow.quadratic_root_divmod(polyr, rini, rtol=rtol, atol=atol)
    r0 = polyr.roots()
    i = np.abs(r0 - r).argmin()
    nt.assert_allclose(r0[i], r, rtol=rtol, atol=atol)
