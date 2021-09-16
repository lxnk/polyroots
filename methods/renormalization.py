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


def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True):
    assert actual.k == desired.k
    nt.assert_array_almost_equal_nulp(actual.r, desired.r, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg,
                                      verbose=verbose)
    nt.assert_array_almost_equal_nulp(actual.p, desired.p, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg,
                                      verbose=verbose)


def asrnumber(c: complex = 1):
    if c.imag == 0:
        return rnumber(np.log(np.absolute(c)), 0 if c.real > 0 else 1, 0)
    else:
        return rnumber(np.log(np.absolute(c)), np.angle(c)/np.pi, 0)


def rnumber(r: float = 0, p: float = 0, k: int = 0):
    return Rnumber(np.float64(r), np.float64(p), np.int32(k))


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

    def __init__(self, r: np.floating = 0, p: np.floating = 0, k: np.integer = 0):
        self.r = r
        self.p = p
        self.k = k
        self.p = self.p % 2

    def __repr__(self):
        return f"Rn({self.r}, {self.p}, {self.k})"

    def __format__(self, format_spec):
        if format_spec == '':
            return f"2{str(self.k).translate(self._superscript_mapping)}⋅({self.r} + 𝜄𝜋⋅{self.p})"
        elif format_spec == 'unicode':
            return f"2{str(self.k).translate(self._superscript_mapping)}⋅({self.r} + 𝜄⋅𝜋⋅{self.p})"
        elif format_spec == 'ascii':
            return f"2**{self.k}*({self.r} + i*pi*{self.p})"

    def __str__(self):
        return f"2{str(self.k).translate(self._superscript_mapping)}⋅[{self.r} + 𝜄𝜋⋅{self.p}]"

    @staticmethod
    def _factor(k1: np.integer, k2: np.integer, dk: int = 0):
        k = max(k1, k2) + dk
        a = int(2 ** (k - k1))
        b = int(2 ** (k - k2))
        return k, a, b

    def __mul__(self, other):
        if issubclass(type(other), Rnumber):
            k, a, b = self._factor(self.k, other.k, 1)
            return Rnumber(self.r / a + other.r / b, self.p / a + other.p / b, k)
        else:
            r = np.log(np.absolute(other)) / 2 ** self.k
            p = np.angle(other) / np.pi / 2 ** self.k
            return Rnumber(self.r + r, self.p + p, self.k)

    def _add_renorm(self, other):
        k, a, b = self._factor(self.k, other.k, 0)
        if self.r / a >= other.r / b:
            c = other.r / b - self.r / a + 1j * np.pi * (other.p / b - self.p / a)
            r = self.r / a
            p = self.p / a
        else:
            c = self.r / a - other.r / b + 1j * np.pi * (self.p / a - other.p / b)
            r = other.r / b
            p = other.p / b
        c = 1 + np.exp((2 ** k) * c)
        r += np.log(np.absolute(c)) / 2**k
        p += np.angle(c) / np.pi / 2**k
        p %= 2
        return Rnumber(r, p, k)

    def __add__(self, other):
        if not issubclass(type(other), Rnumber):
            other = asrnumber(other)
        return self._add_renorm(other)

    def __sub__(self, other):
        if not issubclass(type(other), Rnumber):
            other = asrnumber(other)
        return self._add_renorm(-other)
        # , sub=True)

    def __neg__(self):
        return Rnumber(self.r, (self.p + 1/2**self.k) % 2, self.k)

    # def __pos__(self):
    #     return NotImplemented

    def __abs__(self):
        return Rnumber(self.r, 0, self.k)

    def __complex__(self):
        return complex(np.exp(self.r + 1j * np.pi * self.p))

    def __float__(self):
        pp = np.round(self.p)
        nf = np.exp(self.r)
        if pp == 1:
            nf *= -1
        return float(nf)

    def __rmul__(self, other):
        r = np.log(np.absolute(other)) / 2 ** self.k
        p = np.angle(other) / np.pi / 2 ** self.k
        return Rnumber(self.r + r, self.p + p, self.k)

    def __radd__(self, other):
        return asrnumber(other)._add_renorm(self)

    def __rsub__(self, other):
        return asrnumber(other)._add_renorm(-self)
