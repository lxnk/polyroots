# -*- coding: utf-8 -*-

"""
All complex roots at once:

* aberth [1]_
* durand

.. [1] The aberth method is an improvement of the durand method

Split off a complex conjugated root pair from the real-coefficients polynomial:

* bairstow

Estimate all roots at once (possible complex)

* graeffe [2]_

.. [2] The graeffe method is not very accurate

Find a single root:

* householder [3]_
* laguerre [4]_

.. [3] The householder method can be fast, and returns real values unless we explicit start from complex root,
    though it can diverge

.. [4] The laguerre method is also fast, but always converges, what means that it can jump out of the real roots
    manifold and converge to a complex root

Real root isolation / limits

* vincent [5]_
* bounds

.. [5] The vincent method returns the intervals with a single root

"""


# # This hook works for the importing allows to access modules as
#
# from methods import *
#
# aberth.some_function()
# bounds.some_function()

__all__ = ["aberth", "bairstow", "bounds",
           "durand", "graeffe", "householder",
           "laguerre", "schur", "vincent"]

# # This hook works for the importing allows to access modules as
#
# import methods
#
# methods.bounds.some_function()
# methods.graeffe.some_function()

from . import bounds, graeffe, householder, laguerre

# # This importing allows to access modules even if __init__.py is absent
#
# from methods import bounds, bairstow
#
# bounds.some_function()
# bairstow.some_function()
