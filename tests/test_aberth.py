import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods import aberth
import utils


@pytest.fixture(params=[(-3, 4, 4, 2), (1, 2, 3, 1)], ids=["-3442", "1231"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(polyc):
    r = aberth.roots(polyc)
    nt.assert_allclose(np.sort(r), polyc.roots())


@pytest.mark.parametrize("rtol, atol", [(1e-3, 1e-3), (1e-6, 1e-6)], ids=["e-3", "e-7"])
def test_roots_tol(polyc, rtol, atol):
    r = aberth.roots(polyc, rtol=rtol, atol=atol)
    rp = polyc.roots()
    sr = utils.sort_roots(r)
    srp = np.sort_complex(rp)
    nt.assert_allclose(sr, srp, rtol=rtol, atol=atol)