# -*- coding: utf-8 -*-
"""
Descartes' rule of signs
https://en.wikipedia.org/wiki/Descartes%27_rule_of_signs
Read it! You get definite answer is var=1 or var=0.
If, var = 2, then there is 0 or 2 positive roots
If, var = 3, then there is 1 or 3 positive roots
If, var = 3, then there is 0 or 2 or 4 positive roots, etc.

Vincent's and related theorems
https://en.wikipedia.org/wiki/Real-root_isolation#Vincent's_and_related_theorems

Continued fraction method.
https://en.wikipedia.org/wiki/Real-root_isolation#Continued_fraction_method

Bisection method.
https://en.wikipedia.org/wiki/Real-root_isolation#Bisection_method
"""

from numpy.polynomial import Polynomial as Poly
import numpy as np
from methods.bounds import _ratio as coef_ratio


def sign_var_num(p: Poly):
    return sum(coef_ratio(p.coef) < 0)


def root_intervals_cfrac(p: Poly):
    """Continued fraction method
    Starts from intervals [-inf..0] and [0..+inf]"""
    poly_ival = [(p, np.array((1, 0, 0, 1))), (p(Poly([0, -1])), np.array((-1, 0, 0, 1)))]
    isol_ival = []
    sort_ival = lambda a, b: (a, b) if a < b else (b, a)
    # In first iteration division over zero takes place
    np_settings = np.seterr(divide="ignore")
    while poly_ival:
        a, m = poly_ival[-1]
        poly_ival.pop()
        v = sign_var_num(a)
        if v == 0:
            continue
        elif v == 1:
            # In first iteration _here_ division over zero takes place
            isol_ival.append(sort_ival(m[1]/m[3], m[0]/m[2]))
            continue
        s = 1
        b = a(Poly((s,1)))
        w = v - sign_var_num(b)
        if b(0) == 0:
            isol_ival.append(sort_ival((m[0]*s+m[1])/(m[2]*s+m[3]), (m[0]*s+m[1])/(m[2]*s+m[3])))
            b = b // Poly((0,1))
        poly_ival.append((b, (m[0], m[0]*s+m[1], m[2], m[2]*s+m[3])))
        if w == 0:
            pass
        elif w == 1:
            isol_ival.append(sort_ival(m[1]/m[3], (m[0]*s+m[1])/(m[2]*s+m[3])))
        else:
            a = a(Poly((0, s)))
            a.coef = a.coef[::-1]
            a = a(Poly((1, 1)))
            poly_ival.append((a, (m[1], m[0]*s+m[1], m[3], m[2]*s+m[3])))
    # Restore errors setting
    np.seterr(**np_settings)
    return isol_ival


def root_intervals_bisection(p: Poly, iv: list = [(0, 1)]) -> list:
    """Bisection method
    Starts from the interval [0..1]

    On interval [a,b)   p(x) -> p(M(x)),    M(x) = (bx+a)/(x+1) = b+(a-b)/(x+1)
    """
    poly_ival = iv
    isol_ival = []
    while poly_ival:
        a, b = poly_ival[-1]
        poly_ival.pop()
        if p(a) == 0:
            p = p // Poly((a, 1))
            isol_ival.append((a, a))
        q = p(Poly((b, a-b)))
        q.coef = q.coef[::-1]
        v = sign_var_num(q(Poly((1, 1))))
        if v == 1:
            isol_ival.append((a, b))
        elif v > 1:
            poly_ival.append((a, (a+b)/2))
            poly_ival.append(((a+b)/2, b))
    return isol_ival
