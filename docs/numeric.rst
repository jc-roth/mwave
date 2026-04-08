Numeric reference
#################

The :py:mod:`mwave.numeric` module provides a tree-based framework for numerically simulating atom interferometers. Unlike the symbolic module, which yields analytic phase expressions, this module propagates wavefunctions through the interferometer by applying user-supplied beamsplitter and free-evolution functions at each node. This makes it suitable for simulating realistic effects such as finite pulse duration, intensity inhomogeneity, and atom cloud dynamics.

NumericBraggInterferometer class
================================

.. autoclass:: mwave.numeric.NumericBraggInterferometer
   :members:

NumericTreeNode class
=====================

.. autoclass:: mwave.numeric.NumericTreeNode
   :members:
