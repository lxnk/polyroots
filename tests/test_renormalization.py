# -*- coding: utf-8 -*-

from methods.renormalization import Rnumber
from methods import renormalization as rn
import pytest
import numpy.testing as nt
import numpy as np


def test_rnumber():
    c = Rnumber(1.2, 3.5)
    print(type(c))
    print(c)
    print(f"R-number {c}")


def test_mul():
    c1 = Rnumber(1.2, 3.5)
    c2 = Rnumber(0.6, 0.1)
    # print(c1*c2)
    rn.assert_array_almost_equal_nulp(c1 * c2, Rnumber(0.9, 0.8, 1))


def test_sum():
    c1 = Rnumber(1.2, 3.5)
    c2 = Rnumber(0.6, 0.1)
    print()
    print(c1+c2)


def test_rray():
    c1 = Rnumber(1.2, 3.5)
    c2 = Rnumber(0.6, 0.1)
    z = np.array([c1, c2])
    print(z*z)
