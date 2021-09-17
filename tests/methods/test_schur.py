# -*- coding: utf-8 -*-

from .context import *
from methods import schur


@pytest.fixture(params=[(-.6, .2+.3j, .2-.3j, -1+.1j, -1-.1j, .2)], ids=["r-623"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


def test_schur_cohn_test(polyr):
    n = schur.schur_cohn_test(polyr)
    r = np.abs(polyr.roots())
    nt.assert_equal(n, sum(r<1))
