# -*- coding: utf-8 -*-

import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
from methods import durand
from utils import sort_roots

@pytest.fixture(params=[(-3, 4, 4, 2), (1, 2, 3, 1)], ids=["-3442", "1231"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(polyc):
    r = durand.roots(polyc)
    nt.assert_array_almost_equal_nulp(sort_roots(r), sort_roots(polyc.roots()), 7)


@pytest.mark.parametrize("rtol,atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
def test_roots_tol(polyc, rtol, atol):
    r = durand.roots(polyc, rtol=rtol, atol=atol)
    nt.assert_allclose(sort_roots(r),
                       sort_roots(polyc.roots()), rtol=rtol, atol=atol)