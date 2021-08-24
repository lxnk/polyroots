import numpy as np


def sort_complex(r):
    # TODO: still mixes the other roots (1e-3 case)
    delta = np.max(np.abs(r))
    for i, z in enumerate(r):
        delta = min(delta, np.min(np.abs(z - np.delete(r, i))))
    r = np.sort(r)
    for i, rz in enumerate(np.diff(np.real(r))):
        if abs(rz) < delta and \
                np.imag(r[i + 1]) < 0 and \
                np.abs(np.imag(r[i]) + np.imag(r[i + 1])) < delta:
            r[[i, i+1]] = r[[i+1, i]]
    return r
