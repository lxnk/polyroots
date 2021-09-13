# -*- coding: utf-8 -*-

from methods.renormalization import Rnumber, asrnumber
from methods import renormalization as rn
import pytest
import numpy.testing as nt
import numpy as np


def test_rnumber():
    c = Rnumber(1.2, 3.5)
    print(type(c))
    print(c)
    print(f"R-number {c}, {c:unicode}, {c:ascii}")


def test_asrnumber():
    rc = asrnumber(72.31j)
    # print(rc)
    nt.assert_array_almost_equal_nulp(rc.r, np.log(7**2+2**2)/2)
    nt.assert_array_almost_equal_nulp(rc.p, np.arctan(2/7)/np.pi)


def test_mul():
    c1 = Rnumber(1.4, 3.4, 1)
    c2 = Rnumber(0.5, 0.1, 2)
    c3 = Rnumber(0.6, 0.4, 3)
    # print()
    # # print(complex(c3))
    # print(c1, complex(c1)**2, c2, complex(c2)**4, c3, complex(c3))
    # print(c1*c2, complex(c1)**2 * complex(c2)**4, complex(c1*c2)**8)
    rn.assert_array_almost_equal_nulp(c1 * c2, c3)
    nt.assert_array_almost_equal_nulp(complex(c1)**2 * complex(c2)**4, complex(c1*c2)**8, nulp=10)


def test_number():
    c1 = Rnumber(1.2, 3.5, 2)
    nt.assert_array_almost_equal_nulp(float(c1), np.exp(1.2))
    nt.assert_array_almost_equal_nulp(complex(c1), np.exp(1.2 + 1j*np.pi*1.5))


def test_sum():
    # TODO: sum does not work - check
    c1 = Rnumber(1.4, 3.4, 0)
    c2 = Rnumber(0.5, 0.1, 0)
    print(c1+c2)
    print(complex(c1+c2))
    print(complex(c1), complex(c2))


def test_rray():
    c1 = Rnumber(1.4, 3.4, 1)
    c2 = Rnumber(0.5, 0.1, 2)
    z = np.array([c1, c2])
    print(z*z)
