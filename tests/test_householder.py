# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from tests.context import *
from methods import householder

_coef = [(-3, 4, 4, 2), (1, 2, 3, 1)]
_rini = [(0, -1+1j, -1-1j), (-2, -1+1j, -1-1j)]
_ids = ["c-3442", "c1231"]


@pytest.fixture
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.mark.parametrize("polyc, rini", zip(_coef, _rini), indirect=["polyc"], ids=_ids)
def test_roots(polyc, rini):
    r = householder.roots(polyc, rini)
    # print('\n', sort_roots(r), sort_roots(polyc.roots()))
    nt.assert_allclose(sort_roots(r), sort_roots(polyc.roots()))


@pytest.mark.parametrize("rtol, atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
@pytest.mark.parametrize("polyc, rini", zip(_coef, _rini), indirect=["polyc"], ids=_ids)
def test_roots_tol(polyc, rini, rtol, atol):
    r = householder.roots(polyc, rini, rtol=rtol, atol=atol)
    nt.assert_allclose(sort_roots(r), sort_roots(polyc.roots()), rtol=rtol, atol=atol)