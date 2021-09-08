from numpy.polynomial import Polynomial as Poly
import numpy as np


with open('polydata.npy', 'rb') as f:
    p2 = np.load(f, allow_pickle=True)
    p3 = np.load(f, allow_pickle=True)
    p4 = np.load(f, allow_pickle=True)
    p5 = np.load(f, allow_pickle=True)
    p6 = np.load(f, allow_pickle=True)

print(len(p2), len(p2['coef']),
      len(p2[7]), len(p2[7]['coef']), len(p2['coef'][7]), len(p2['root'][7]))

print(len(p5), len(p5['coef']),
      len(p5[7]), len(p5[7]['coef']), len(p5['coef'][7]), len(p5['root'][7]))

print(len(p2), len(p3), len(p4), len(p5), len(p6))

pl = Poly(p4[7]['coef'])
print(pl)
print(pl.roots())
print(p4[7]['root'])
