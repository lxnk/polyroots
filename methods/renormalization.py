# -*- coding: utf-8 -*-
"""
Numeric type to support Malajovich–Zubelli renormalisation
https://en.wikipedia.org/wiki/Graeffe%27s_method#Renormalization
"""

import numpy as np
import numpy.testing as nt


def assert_array_almost_equal_nulp(x, y, nulp=1):
    assert x.k == y.k
    nt.assert_array_almost_equal_nulp(x.r, y.r, nulp=nulp)
    nt.assert_array_almost_equal_nulp(x.p, y.p, nulp=nulp)


def assert_array_almost_equal_nulp(x, y, nulp=1):
    assert x.k == y.k
    nt.assert_array_almost_equal_nulp(x.r, y.r, nulp=nulp)
    nt.assert_array_almost_equal_nulp(x.p, y.p, nulp=nulp)


def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True):
    assert actual.k == desired.k
    nt.assert_array_almost_equal_nulp(actual.r, desired.r, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg,
                                      verbose=verbose)
    nt.assert_array_almost_equal_nulp(actual.p, desired.p, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg,
                                      verbose=verbose)


class Rnumber:
    """A numeric class that represent the number in an exponential form

    .. math:: c = exp(2^k (r+i\\pi p)), \\quad \\forall r \\in \\mathcal{R}, \
        \\quad \\forall p \\in [0..2), \\quad \\forall k \\in \\mathcal{Z}

    """
    r: np.floating = 0
    p: np.floating = 0
    k: np.integer = 0

    # Unicode character mappings for improved __str__
    _superscript_mapping = str.maketrans({
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    })

    def __init__(self, r: float = 0, p: float = 0, k: float = 0):
        self.r = np.float64(r)
        self.p = np.float64(p)
        self.k = np.int32(k)
        self.p = self.p % 2

    def __repr__(self):
        return f"Rnum({self.r}, {self.p}, {self.k})"

    def __format__(self, format_spec):
        return f"2{str(self.k).translate(self._superscript_mapping)}⋅({self.r} + 𝜄𝜋⋅{self.p})"
        # return f"2{self.k}⋅({self.r} + 𝜄𝜋⋅{self.p})"

    def __str__(self):
        return f"2{str(self.k).translate(self._superscript_mapping)}⋅({self.r} + 𝜄𝜋⋅{self.p})"
        # return f"2**{self.k} * ({self.r} + i*pi * {self.p} )"

    @staticmethod
    def _factor(k1: np.integer, k2: np.integer, dk: int = 0):
        k = max(k1, k2) + dk
        a = int(2 ** (k - k1))
        b = int(2 ** (k - k2))
        return k, a, b

    def __mul__(self, other):
        k, a, b = self._factor(self.k, other.k, 1)
        return Rnumber(self.r / a + other.r / b, self.p / a + other.p / b, k)

    def _add_renorm(self, other, sub: bool = False):
        k, a, b = self._factor(self.k, other.k, 0)
        if sub:
            dp = 1
        else:
            dp = 0
        if self.r / a >= other.r / b:
            c = other.r / b - self.r / a + 1j * np.pi * ((other.p + dp) / b - self.p / a)
            r = self.r
            p = self.p
        else:
            c = self.r / a - other.r / b + 1j * np.pi * (self.p / a - (other.p + dp) / b)
            r = other.r
            p = other.p
        c = 1 + np.exp(2 ** k * c)
        r += np.log(np.absolute(c))
        p += np.angle(c) / np.pi
        return Rnumber(r, p, k)

    def __add__(self, other):
        return self._add_renorm(other)

    def __sub__(self, other):
        return self._add_renorm(other, sub=True)
