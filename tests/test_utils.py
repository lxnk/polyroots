# -*- coding: utf-8 -*-

from . import *
from utils import sort_roots
import numpy as np


@pytest.mark.parametrize("ri,rs", [([1+1j], [1+1j]),
                                   ([1+1j, 1.001-1j, 2], [1.001 - 1j, 1 + 1j, 2])], ids=["r1", "r1c2"])
def test_sort_roots(ri, rs):
    # nt.assert_equal(np.sort(r), [1+1j, 1.001-1j, 2])
    nt.assert_equal(sort_roots(ri), rs)


