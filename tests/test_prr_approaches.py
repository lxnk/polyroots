from . import *
from utils import sort_roots
from utils import unique as unique_in_sort
import numpy as np
from methods import householder, bounds, vincent, graeffe, laguerre


# @pytest.fixture(params=[(3, 5, 5, 2000, -5, 3)])
@pytest.fixture(params=[(-3, 4, 4, 2),
                        (1, 2, 3, 1),
                        (.25, .5, .751, .9),
                        (3, 5, 5, 2000, 2),
                        (3, 5, 5, 2000, -5, 3),
                        (3, 5, -50, 2000, 70, 3),
                        (3, -5, -5, 2, 7, 3)])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


# @pytest.mark.skip
# def test_positive_real_roots_decartes(polyc):
#     rc = polyc.roots()
#     print('\n', rc)
#     rr = rc[(rc.imag == 0) & (rc.real >= 0)].real
#     p = polyc.copy()
#     print("n0=", vincent.sign_var_num(p))
#     r = list()
#     while vincent.sign_var_num(p) > 0:
#         x = bounds.root_limit(p, method="lagrange", rproots=True)
#         print("x0=", x)
#         x = householder.roots(p, x, d=1)
#         print("x=", x)
#         r.append(x)
#         p = p // [-x, 1]
#         print("p=", p)
#         print("n=", vincent.sign_var_num(p))
#     print(r, rr)
#     # nt.assert_allclose(np.sort(r), np.sort(rr))


def test_positive_real_roots_graeffe_householder(polyc):
    rc = polyc.roots()
    # print('\n', rc)
    rr = rc[(rc.imag == 0) & (rc.real >= 0)].real
    r = graeffe.roots_classical(polyc, d=0, absval=True)
    r = householder.roots(polyc, r, d=1)
    r = np.unique(r[r >= 0])
    # print(r, rr)
    nt.assert_allclose(np.sort(r), np.sort(rr))


def test_positive_real_roots_graeffe_laguerre(polyc):
    rc = polyc.roots()
    # print('\n', rc)
    rr = rc[(rc.imag == 0) & (rc.real >= 0)].real
    r = graeffe.roots_classical(polyc, d=0, absval=True)
    r = laguerre.roots(polyc, r)
    r = np.unique(r[(r.imag == 0) & (r.real >= 0)].real)
    # print(r, rr)
    nt.assert_allclose(np.sort(r), np.sort(rr))


# @pytest.mark.skip
# def test_positive_real_roots_graeffe_db():
#     with open('data/polydata.npy', 'rb') as f:
#         poly2 = np.load(f, allow_pickle=True)
#         poly3 = np.load(f, allow_pickle=True)
#         poly4 = np.load(f, allow_pickle=True)
#         poly5 = np.load(f, allow_pickle=True)
#         poly6 = np.load(f, allow_pickle=True)
#     # print(len(poly2), len(poly2['coef']),
#     #       len(poly2[7]), len(poly2[7]['coef']), len(poly2['coef'][7]), len(poly2['root'][7]))
#     # print(len(poly5), len(poly5['coef']), len(poly5['root']),
#     #       len(poly5[7]), len(poly5[7]['coef']), len(poly5['coef'][7]), len(poly5['root'][7]))
#
#     # print()
#     i = 0
#     for pl in poly5:
#         p = Poly(pl['coef'])
#         r0 = sort_roots(pl['root'])
#         r1 = sort_roots(p.roots())
#         # print(r0)
#         # print(r)
#         nt.assert_allclose(r1, r0, rtol=2e-5, atol=2e-5)
#         rr = np.sort(np.unique(r1[(r1.imag == 0) & (r1.real >= 0)].real))
#         r = graeffe.roots_classical(p, d=0, absval=True)
#         rmax = bounds.root_limit(p, method="lagrange", rproots=True)
#         r = np.append(r[r < rmax], rmax)
#         # r = householder.roots(p, r, d=1)
#         r = laguerre.roots(p, r, rtol=1e-6, atol=1e-6)
#         r = unique_in_sort(np.sort(r[r >= 0]), rtol=1e-6, atol=1e-6)
#         print("r  =", r)
#         print("rr =", rr)
#         if len(r) != len(rr):
#             print("i=", i)
#         i += 1
