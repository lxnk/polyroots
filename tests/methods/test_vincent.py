# -*- coding: utf-8 -*-

from .context import *
from methods import vincent


@pytest.fixture
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


@pytest.fixture(params=[(-.6, -.3, .49, .52, .21),
                        (.25, .5, .75, .9)], ids=["r-3432", "rquad"])
def polyr(request):
    """Create polynomial out of the real roots"""
    return Poly(pl.polyfromroots(request.param), domain=[0, 1], window=[0, 1])


@pytest.mark.parametrize("polyc, num", [((-3, 4, -4, 2), 3)], indirect=["polyc"], ids=["c-+-+"])
def test_sign_var_num(polyc, num):
    n = vincent.sign_var_num(polyc)
    nt.assert_equal(n, num)


def test_root_intervals_cfrac(polyr):
    """Last data
    [(-1, -1/2), (-1/2, 0.0), (1/2, 1), (1/3, 1/2), (0.0, 1/3)]
    [(3/4, 4/5), (0, 1), (1/3, 1/2), (0, 1/3)]
    """
    iv = vincent.root_intervals_cfrac(polyr)
    r = polyr.roots()
    # print('\n', iv, '<->', r)
    for a, b in iv:
        nt.assert_equal(len(r[(r > a) & (r <= b)]), 1)


def test_root_intervals_bisection(polyr):
    """Last data
    [(1/2, 1), (1/4, 1/2), (0, 1/4)]
    [(7/8, 1), (3/4, 7/8), (3/8, 1/2), (1/4, 3/8)]
    TODO: check hitting the values
    """
    iv = vincent.root_intervals_bisection(polyr, [(-1, 1)])
    r = polyr.roots()
    # print('\n', iv, '<->', polyr.roots())
    for a, b in iv:
        nt.assert_equal(len(r[(r > a) & (r <= b)]), 1)


def test_polyscale():
    # a, b = (235.8206580766257, 2424733.74139227)
    # p = Poly([7.868426686580592e-12, -4.2171839126289425e-09, 4.420539641958822e-07,
    #           4.066178974384906e-05, -0.001772910664260689, -6.938893903907228e-18])

    a, b = (0.2225881992598322, 33769190.08343536)
    # 0.057284547625225925
    p = Poly([4.236486426313966e-05, -0.001592757183370336, 0.012910136781970094,
              0.053141754849931735, -0.26057129509954186, 5.551115123125783e-17])

    rp = p.roots()
    print()
    print(rp[rp.imag == 0].real)
    q = p(Poly((b, a - b)))
    q.coef = q.coef[::-1]
    q = q(Poly((1, 1)))
    rq = q.roots()
    pa = p(Poly((a, b-a)))
    rpa = pa.roots()
    print(rq[rq.imag == 0].real, rpa[rpa.imag == 0].real)

    pa.coef = pa.coef[::-1]
    pa = pa(Poly((1, 1)))
    rpa = pa.roots()
    print(rq[rq.imag == 0].real, rpa[rpa.imag == 0].real)
