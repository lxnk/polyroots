# -*- coding: utf-8 -*-
"""Continued fraction method.
https://en.wikipedia.org/wiki/Real-root_isolation#Continued_fraction_method
"""
from numpy.polynomial import Polynomial as Poly
import numpy as np
import utils


def sign_var_num(p: Poly):
    c = p.coef[p.coef != 0]
    return sum(c[:-1]*c[1:] < 0)


def root_intervals(p: Poly):
    poly_ival = [(p, np.array((1, 0, 0, 1))), (p(Poly([0, -1])), np.array((-1, 0, 0, 1)))]
    isol_ival = []
    # utils.show_roots(np.array([0, 6]), p)
    while poly_ival:
        a, m = poly_ival[-1]
        poly_ival.pop()
        v = sign_var_num(a)
        if v == 0:
            continue
        elif v == 1:
            isol_ival.append((m[1]/m[3], m[0]/m[2]))
            continue
        s = 1
        b = a(Poly((s,1)))
        # utils.show_roots(np.array([0, 6]), b)
        w = v - sign_var_num(b)
        if b(0) == 0:
            isol_ival.append(((m[0]*s+m[1])/(m[2]*s+m[3]), (m[0]*s+m[1])/(m[2]*s+m[3])))
            b = b // Poly((0,1))
        poly_ival.append((b,(m[0], m[0]*s+m[1], m[2], m[2]*s+m[3])))
        if w == 0:
            pass
        elif w == 1:
            isol_ival.append((m[1]/m[3], (m[0]*s+m[1])/(m[2]*s+m[3])))
        else:
            a = a(Poly((0, s)))
            a.coef = a.coef[::-1]
            a = a(Poly((1, 1)))
            poly_ival.append((a, (m[1], m[0]*s+m[1], m[3], m[2]*s+m[3])))
    return isol_ival