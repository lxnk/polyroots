import numpy as np

x = np.zeros(11)

# rng = np.random.default_rng(82749)

with open('../data/polydata.npy', 'rb') as f:
    for d in range(2, 7):
        p = np.load(f)
        pc = p['coef']
        print(f"{len(pc)} polynomials of degree {d}")

        if len(pc) > 128:
            # print(rng.integers(low=0, high=len(p['coef']), size=128))
            pc = pc[np.arange(0, len(pc), 9)]
            print(f'take {len(pc)}')

        with open(f'poly{d}d.csv', 'w') as f2:
            for pp in pc:
                x[:] = 0
                x[0:d+1] = pp

                s = np.array2string(x, max_line_width=np.inf, separator=',',
                                    formatter={'float_kind':lambda x: "%.14g" % x})
                f2.write(f'{s};{d};// {np.max(x.nonzero())}th degree\n')


