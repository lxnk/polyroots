import numpy as np


def sort_roots(r):
    delta = np.max(np.abs(r))
    for i, z in enumerate(r):
        delta = min(delta, np.min(np.abs(z - np.delete(r, i))))
    r = np.sort(r)
    for i, _ in enumerate(r[::-2]):
        if np.abs(r[i]-r[i+1].conj()) < delta and \
                r[i+1].imag < 0:
            r[[i, i+1]] = r[[i+1, i]]
    return r


def tol(t, rtol=0, atol=0):
    return np.maximum(atol + rtol * t, np.spacing(t))