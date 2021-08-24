import pytest
import numpy.testing as nt
from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods import durand

poly_coef = {(-3, 4, 4, 2), (1, 2, 3, 1)}
#
#
# @pytest.mark.parametrize("pc", poly_coef)
# def setup(pc):
#     """Module-level setup"""
#     poly = Poly(pc, domain=[0, 1], window=[0, 1])
#     print('doing setup')


@pytest.mark.parametrize("pc", poly_coef)
def test_roots(pc):
    p = Poly(pc, domain=[0, 1], window=[0, 1])
    r = durand.roots(p)
    nt.assert_allclose(r, p.roots())
