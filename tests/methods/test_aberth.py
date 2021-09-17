# -*- coding: utf-8 -*-

from .context import *
from methods import aberth


@pytest.fixture(params=[(-3, 4, 4, 2), (1, 2, 3, 1)], ids=["-3442", "1231"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(polyc):
    r = aberth.roots(polyc)
    nt.assert_allclose(sort_roots(r), sort_roots(polyc.roots()))


@pytest.mark.parametrize("rtol, atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
def test_roots_tol(polyc, rtol, atol):
    r = aberth.roots(polyc, rtol=rtol, atol=atol)
    nt.assert_allclose(sort_roots(r), sort_roots(polyc.roots()), rtol=rtol, atol=atol)