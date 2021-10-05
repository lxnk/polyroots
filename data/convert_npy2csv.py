import numpy as np

x = np.zeros(11)

with open('../data/polydata.npy', 'rb') as f:
    for d in range(2, 7):
        p = np.load(f)
        print(f"{len(p['coef'])} polynomials of degree {d}")

        with open(f'poly{d}d.csv', 'w') as f2:
            for pp in p['coef']:
                x[:] = 0
                x[0:d+1] = pp
                s = np.array2string(x, max_line_width=np.inf, separator=',',
                                    formatter={'float_kind':lambda x: "%.14g" % x})
                f2.write(f'{s};{d};// {np.max(x.nonzero())}th degree\n')


