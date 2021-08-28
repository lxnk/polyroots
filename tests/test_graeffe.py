import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial import polynomial as pl
import numpy as np
from methods import graeffe
import utils


@pytest.fixture(params=[(-3, 4, 4, 2)], ids=["c-3442"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])

@pytest.fixture(params=[(-3, 4, 3, 2)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the coefficients"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])

def test_dgiteration(polyc):
    pp = polyc.copy()
    pp.coef[1::2] *= -1
    pp = pp * polyc
    pg = graeffe.dgiteration(polyc)
    nt.assert_array_equal(pp.coef[::2], pg.coef)


def test_roots_classical(polyc):
    polyc.coef /= polyc.coef[polyc.degree()]
    r = graeffe.roots_classical(polyc, 7)
    print('\n', utils.sort_roots(r),
          '\n', np.sort_complex(polyc.roots()))
    # nt.assert_allclose(utils.sort_roots(r),
    #                    np.sort_complex(poly.roots()), rtol=1e-2)


def test_roots_classical2(polyr):
    polyr.coef /= polyr.coef[polyr.degree()]
    r = graeffe.roots_classical(polyr, 7)
    print('\n', utils.sort_roots(r),
          '\n', np.sort_complex(polyr.roots()))
    # nt.assert_allclose(utils.sort_roots(r),
    #                    np.sort_complex(poly.roots()), rtol=1e-2)


def test_roots_tangential(polyc):
    polyc.coef /= polyc.coef[polyc.degree()]
    r = graeffe.roots_tangential(polyc, 5, 1e-2 * (1 + 0.1j))
    print('\n', utils.sort_roots(r),
          '\n', np.sort_complex(polyc.roots()))
    # nt.assert_allclose(utils.sort_roots(r),
    #                    np.sort_complex(poly.roots()), rtol=1e-2)


