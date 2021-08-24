from methods import householder
from methods import durand
import numpy as np
from numpy.polynomial import Polynomial as Poly


p = Poly([-3, 4, 4, 2], domain=[0, 1], window=[0, 1])

# print(p)
r = durand.roots(p)
print(r)
print(p.roots())
# print(p.coef[p.degree()])
# print([p(1/2), p(-3/2)])

# r = householder.roots(p, 1, 1)
# r = householder.roots(p, 1, 5)
#
# r = householder.roots(p, -1, 1)
# r = householder.roots(p, -1, 5)


# print(r)
# print(p.coef)
# print(p.domain)
# print(p.window)
# print(c)
