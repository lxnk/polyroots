from numpy.polynomial import Polynomial as Poly
import numpy as np


def access_data():
    p = dict()
    with open('../data/polydata.npy', 'rb') as f:
        for d in range(2, 7):
            p[d] = np.load(f)
    return p


# p2 = np.load(f, allow_pickle=True, mmap_mode='r')
# p3 = np.load(f, allow_pickle=True, mmap_mode='r')
# p4 = np.load(f, allow_pickle=True, mmap_mode='r')
# p5 = np.load(f, allow_pickle=True, mmap_mode='r')
# p6 = np.load(f, allow_pickle=True, mmap_mode='r')

p = access_data()


print(len(p[2]), len(p[2]['coef']),
      len(p[2][7]), len(p[2][7]['coef']), len(p[2]['coef'][7]), len(p[2]['root'][7]))

# print(len(p2), len(p2['coef']),
#       len(p2[7]), len(p2[7]['coef']), len(p2['coef'][7]), len(p2['root'][7]))

# print(len(p5), len(p5['coef']),
#       len(p5[7]), len(p5[7]['coef']), len(p5['coef'][7]), len(p5['root'][7]))
#
# print(len(p2), len(p3), len(p4), len(p5), len(p6))
#
# pl = Poly(p4[7]['coef'])
# print(pl)
# print(pl.roots())
# print(p4[7]['root'])
