import numpy as np
from itertools import tee


def sort_roots(r):
    # TODO: still mixes the other roots (1e-3 case)
    delta = np.max(np.abs(r))
    for i, z in enumerate(r):
        delta = min(delta, np.min(np.abs(z - np.delete(r, i))))
    r = np.sort(r)
    for i, _ in enumerate(r[::-2]):
        if np.abs(r[i]-r[i+1].conj()) < delta and \
                np.imag(r[i+1]) < 0:
            r[[i, i+1]] = r[[i+1, i]]
    return r
