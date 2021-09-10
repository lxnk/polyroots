# -*- coding: utf-8 -*-

from methods.renormalization import Rnumber


def test_rnumber():
    c1 = Rnumber(1.2, 3.5)
    c2 = Rnumber(0.6, 0.1)
    print()
    print(c1*c2)
    print(Rnumber(0.9, 0.8, 1))
    assert c1*c2 == Rnumber(0.9, 0.8, 1)
