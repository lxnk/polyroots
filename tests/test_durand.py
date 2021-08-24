import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods import durand
import util


@pytest.fixture(params=[(-3, 4, 4, 2), (1, 2, 3, 1)], ids=["-3442", "1231"])
def poly(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(poly):
    r = durand.roots(poly)
    nt.assert_allclose(r, poly.roots())


@pytest.mark.parametrize("rtol,atol", [(1e-6, 1e-6), (1e-7, 1e-7)], ids=["e-3", "e-7"])
def test_roots_tol(poly, rtol, atol):
    r = durand.roots(poly, rtol=rtol, atol=atol)
    rp = poly.roots()
    sr = np.sort_complex(r)
    srp = util.sort_complex(rp)
    print('\n')
    print(sr, np.real(sr[0]-sr[1]), np.imag(sr[0]+sr[1]))
    print(srp, np.real(srp[0]-sr[1]), np.imag(srp[0]+srp[1]))
    nt.assert_allclose(sr, srp, rtol=rtol, atol=atol)