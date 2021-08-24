import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods import aberth
import utils


@pytest.fixture(params=[(-3, 4, 4, 2), (1, 2, 3, 1)], ids=["-3442", "1231"])
def poly(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(poly):
    r = aberth.roots(poly)
    nt.assert_allclose(np.sort(r), poly.roots())


@pytest.mark.parametrize("rtol,atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
def test_roots_tol(poly, rtol, atol):
    r = aberth.roots(poly, rtol=rtol, atol=atol)
    rp = poly.roots()
    sr = utils.sort_roots(r)
    srp = np.sort_complex(rp)
    nt.assert_allclose(sr, srp, rtol=rtol, atol=atol)