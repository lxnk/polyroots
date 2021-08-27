import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods import graeffe
import utils


@pytest.fixture(params=[(-3, 4, 4, 2)], ids=["-3442"])
def poly(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_dgiteration(poly):
    pp = poly.copy()
    pp.coef[1::2] *= -1
    pp = pp * poly
    pg = graeffe.dgiteration(poly)
    nt.assert_array_equal(pp.coef[::2], pg.coef)


def test_roots_classical(poly):
    poly.coef /= poly.coef[poly.degree()]
    r = graeffe.roots_classical(poly, 5)
    print('\n', utils.sort_roots(r),
          '\n', np.sort_complex(poly.roots()))
    # nt.assert_allclose(utils.sort_roots(r),
    #                    np.sort_complex(poly.roots()), rtol=1e-2)


def test_roots_tangential(poly):
    poly.coef /= poly.coef[poly.degree()]
    r = graeffe.roots_tangential(poly, 5, 1e-2*(1+0.1j))
    print('\n', utils.sort_roots(r),
          '\n', np.sort_complex(poly.roots()))
    # nt.assert_allclose(utils.sort_roots(r),
    #                    np.sort_complex(poly.roots()), rtol=1e-2)


