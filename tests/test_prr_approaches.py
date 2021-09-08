from . import *
from utils import sort_roots
import numpy as np
from methods import householder, bounds, vincent


# @pytest.fixture(params=[(-3, 4, 4, 2),
#                         (1, 2, 3, 1),
#                         (.25, .5, .751, .9),
#                         (3, 5, 5, 2000, 2),
#                         (3, 5, 5, 2000, -5, 3),
#                         (3, 5, -50, 2000, 70, 3)])
@pytest.fixture(params=[(3, 5, 5, 2000, -5, 3)])
def polyc(request):
    """Create polynomial out of the coefficients"""
    return Poly(request.param, domain=[0, 1], window=[0, 1])


def test_roots(polyc):
    rc = polyc.roots()
    print('\n', rc)
    rr = rc[(rc.imag == 0) & (rc.real >= 0)].real
    p = polyc.copy()
    print("n0=", vincent.sign_var_num(p))
    r = list()
    while vincent.sign_var_num(p) > 0:
        x = bounds.root_limit(p, method="lagrange", rproots=True)
        print("x0=", x)
        x = householder.roots(p, x, d=1)
        print("x=", x)
        r.append(x)
        p = p // [-x, 1]
        print("p=", p)
        print("n=", vincent.sign_var_num(p))
    print(r, rr)
    # nt.assert_allclose(np.sort(r), np.sort(rr))