# -*- coding: utf-8 -*-

from . import *
from numpy.polynomial import polynomial as pl
import numpy as np
from methods import graeffe
from utils import sort_roots


@pytest.fixture(params=[(-3, 4, -4, 2)], ids=["c-3442"])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-3, 4, 5, 2)], ids=["r-3432"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


def test_dgiteration(polyc):
    pp = polyc.copy()
    pp.coef[1::2] *= -1
    pp = pp * polyc
    pg = graeffe.dgiteration(polyc)
    nt.assert_array_equal(pp.coef[::2], pg.coef)


def test_roots_classical_real_roots(polyr):
    polyr.coef /= polyr.coef[polyr.degree()]
    r = graeffe.roots_classical(polyr, 3)
    # print('\n', sort_roots(r), '\n', sort_roots(polyr.roots()))
    nt.assert_allclose(sort_roots(r), sort_roots(polyr.roots()), rtol=3e-2)


def test_roots_tangential(polyr):
    polyr.coef /= polyr.coef[polyr.degree()]
    r = graeffe.roots_tangential(polyr, 5, 1e-3)
    # print('\n', sort_roots(r), sort_roots(polyr.roots()))
    nt.assert_allclose(sort_roots(r), sort_roots(polyr.roots()), rtol=3e-2)
