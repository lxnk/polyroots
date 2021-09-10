# -*- coding: utf-8 -*-
"""Dandelin–Lobachesky–Graeffe method.
https://en.wikipedia.org/wiki/Graeffe%27s_method
"""

from numpy.polynomial import Polynomial as Poly


def stransform(p: Poly) -> Poly:
    """Schur transform
    For a complex polynomial `p(z)` of degree `n` its reciprocal adjoint polynomial :math:`p_r(z)` is defined as

    .. math::
        p(z)   = \\sum_{k=0} ^n a_k z^k

    .. math::
        p_r(z) = z^n [p(1/z^*)]^* = \\sum_{k=0} ^n a_{n-k}^* z^k

    and its Schur Transform `Tp` by

    .. math::
        Tp(z)   = p(0)^* p(z) - p_r(0)^* p_r(z)

    Star means here a complex conjugation.
    """
    return Poly(p.coef[0].conj() * p.coef[:-1] - p.coef[-1] * p.coef[:0:-1].conj())


def schur_cohn_test(p: Poly) -> int:
    """See Theorem [Schur-Cohn test] in
    https://en.wikipedia.org/wiki/Lehmer–Schur_algorithm

    Let `p` be a complex polynomial with :math:`Tp\\neq 0` and let K be the smallest number such that
    :math:`T^{K+1}p=0`.
    Moreover let :math:`\\delta_{k}=(T^{k}p)(0)` for `k=1..K` and :math:`d_{k}=\\deg T^{k}p` for `k=0..K`.

    *  All roots of `p` lie inside the unit circle if and only if :math:`\\delta_{1}<0`, :math:`\\delta_{k}>0` for
       `k=2,..K`,and :math:`d_{K}=0`.

    *  All roots of `p` lie outside the unit circle if and only if :math:`\\delta _{k}>0` for `k=1..K` and
       :math:`d_{K}=0`.


    *   If :math:`d_{K}=0` and if :math:`\\delta _{k}<0` for :math:`k=k_{0},k_{1},..k_{m}` (in increasing order) and
        :math:`\\delta _{k}>0` otherwise, then `p` has no roots on the unit circle and the number of roots of `p`
        inside the unit circle is :math:`\\sum_{i=0}^{m}(-1)^{i}d_{k_{i}-1}`.


    The Schur-Cohn test can be applied to the polynomial `q` only if none of the following equalities occur:
    :math:`T^{k}q(0)=0` for some `k=1..K` or :math:`T^{K+1}q=0` while :math:`d_{K}>0`. Note that in both cases
    :math:`\\delta` is zero.
    """
    num = 0
    sgn = 1
    d = p.degree()
    while d > 0:
        p = stransform(p)
        dt = p(0)
        assert dt != 0
        if dt < 0:
            num += sgn * d
            sgn = -sgn
        d = p.degree()
    return num
