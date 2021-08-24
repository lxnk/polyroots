
import numpy.testing as nt
import numpy as np
import util


def test_sort_complex():
    r = [1+1j, 1.001-1j, 2]
    # rs = util.sort_complex(r)
    nt.assert_equal(np.sort(r), [1+1j, 1.001-1j, 2])
    nt.assert_equal(util.sort_complex(r), [1.001-1j, 1+1j, 2])
