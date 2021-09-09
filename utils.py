import numpy as np
import matplotlib.pyplot as plt


def sort_roots(r):
    if len(r) > 1:
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


def show_roots(r, p):
    _, ax = plt.subplots()
    xmin, xmax = (min(r.real), max(r.real))
    xmin -= 0.1 * (xmax - xmin)
    xmax += 0.1 * (xmax - xmin)
    x = np.linspace(xmin, xmax)
    ax.plot(x, p(x), '-')
    ax.plot(r.real, np.real(p(r)), 'x')
    for z in r:
        if z.imag != 0:
            x = np.linspace(0, 1.2*z.imag)
            ax.plot(x+z.real, np.real(p(z.real+x*1j)), ':',
                    x+z.real, np.imag(p(z.real+x*1j)), ':')
        ax.plot(z.imag+z.real, 0, '+')
    ax.spines['bottom'].set_position('zero')
    ax.set_yscale('linear')
    plt.show()


def unique(a: np.array, rtol: float = 0, atol: float = 0):
    test = np.abs(np.diff(a, prepend=np.inf)) >= tol(a, rtol=rtol, atol=atol)
    return a[test]