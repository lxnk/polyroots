# -*- coding: utf-8 -*-
"""
Numeric type to support Malajovich–Zubelli renormalisation
https://en.wikipedia.org/wiki/Graeffe%27s_method#Renormalization
"""

import numpy as np


class Rnumber:
    """A numeric class that represent the number in an exponential form

    .. math:: c = exp(2^k (r+i\\pi p)), \\quad \\forall r \\in \\mathcal{R}, \
        \\quad \\forall p \\in [0..2), \\quad \\forall k \\in \\mathcal{Z}

    """
    r: np.floating = 0
    p: np.floating = 0
    k: np.integer = 0

    def __init__(self, r: np.floating = 0, p: np.floating = 0, k: np.integer = 0):
        self.r = r
        self.p = p
        self.k = k
        self.p = self.p % 2

    def __repr__(self):
        return "Rnumber"

    def __str__(self):
        return f"2^{self.k} * ( {self.r} + pi * {self.p} )"

    def __mul__(self, other):
        k = max(self.k, other.k) + 1
        a = 2**(self.k-k)
        b = 2**(other.k-k)
        return Rnumber(a*self.r + b*other.r, a*self.p + b*other.p, k)
