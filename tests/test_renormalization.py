# -*- coding: utf-8 -*-

from methods.renormalization import rnumber, asrnumber
from methods import renormalization as rn
import pytest
import numpy.testing as nt
import numpy as np


def test_rnumber():
    c = rnumber(1.2, 3.5)
    # print(type(c))
    # print(c)
    # print(f"R-number {c}, {c:unicode}, {c:ascii}")
    nt.assert_string_equal(str(c), "2⁰⋅[1.2 + 𝜄𝜋⋅1.5]")
    nt.assert_string_equal(f"{c}", "2⁰⋅(1.2 + 𝜄𝜋⋅1.5)")
    nt.assert_string_equal(f"{c:unicode}", "2⁰⋅(1.2 + 𝜄⋅𝜋⋅1.5)")
    nt.assert_string_equal(f"{c:ascii}", "2**0*(1.2 + i*pi*1.5)")


def test_asrnumber():
    rc = asrnumber(7 + 2j)
    nt.assert_array_almost_equal_nulp(rc.r, np.log(7**2+2**2)/2)
    nt.assert_array_almost_equal_nulp(rc.p, np.arctan(2/7)/np.pi)
    rc = asrnumber(7 - 2j)
    nt.assert_array_almost_equal_nulp(rc.p, 2-np.arctan(2/7)/np.pi)
    rc = asrnumber(-7 - 2j)
    nt.assert_array_almost_equal_nulp(rc.p, 1+np.arctan(2/7)/np.pi)


def test_mul():
    c1 = rnumber(1.4, 3.4, 1)
    c2 = rnumber(0.5, 0.1, 2)
    c3 = rnumber(0.6, 0.4, 3)
    # print()
    # print(c3)
    # print(c1 * c2)
    rn.assert_array_almost_equal_nulp(c1 * c2, c3)
    nt.assert_array_almost_equal_nulp(complex(c1)**2 * complex(c2)**4, complex(c1*c2)**8, nulp=7)


def test_number():
    c1 = rnumber(1.2, 3.5, 1)
    nt.assert_array_almost_equal_nulp(float(c1), np.exp(1.2))
    nt.assert_array_almost_equal_nulp(complex(c1), np.exp(1.2 + 1j*np.pi*1.5))


def test_add():
    # TODO: sum does not work - check
    c1 = rnumber(1.4, 3.4, 1)
    c2 = rnumber(0.5, 0.1, 2)

    # print()
    # print(c3)
    # print(c1 + c2)
    nt.assert_array_almost_equal_nulp(complex(c1+c2)**4, complex(c1)**2 + complex(c2)**4, nulp=7)


def test_sub():
    # TODO: sum does not work - check
    c1 = rnumber(1.4, 3.4, 1)
    c2 = rnumber(0.5, 0.1, 2)
    nt.assert_array_almost_equal_nulp(complex(c1-c2)**4, complex(c1)**2 - complex(c2)**4, nulp=7)


def test_rray():
    c0 = rnumber(1.6, 0.8, 0)
    c1 = rnumber(1.4, 3.4, 1)
    c2 = rnumber(0.5, 0.1, 2)
    z01 = np.array([c0, c1])
    z12 = np.array([c1, c2])
    zz = z01*z12
    rn.assert_array_almost_equal_nulp(zz[0], c0 * c1)
    rn.assert_array_almost_equal_nulp(zz[1], c1 * c2)


def test_rmul():
    c1 = rnumber(1.4, 3.4, 1)
    ac = 3*c1
    ca = c1*3
    # print()
    # print(ac)
    # print(ca)
    nt.assert_array_almost_equal_nulp(complex(ac)**2, 3 * complex(c1)**2, nulp=2)
    nt.assert_array_almost_equal_nulp(complex(ca)**2, 3 * complex(c1)**2, nulp=2)


def test_radd():
    c1 = rnumber(1.4, 3.4, 1)
    ac = c1 + 3
    ca = 3 + c1
    # print()
    # print(ac)
    # print(3+complex(c1)**2)
    # print(complex(ca)**2)
    nt.assert_array_almost_equal_nulp(complex(ac)**2, 3 + complex(c1)**2, nulp=13)
    nt.assert_array_almost_equal_nulp(complex(ca)**2, 3 + complex(c1)**2, nulp=13)


def test_rsub():
    c1 = rnumber(1.4, 3.4, 1)
    ac = c1 - 3
    ca = 3 - c1
    # print()
    # print(c1)
    # print(-c1)
    # print(complex(c1) ** 2)
    # print(complex(-c1) ** 2)
    # print(-3+complex(c1)**2)
    # print(complex(ac)**2)
    # print(complex(ca) ** 2)
    nt.assert_array_almost_equal_nulp(complex(ac)**2, -3 + complex(c1)**2, nulp=13)
    nt.assert_array_almost_equal_nulp(complex(ca)**2, 3 - complex(c1)**2, nulp=13)
