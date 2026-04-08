Symbolic reference
##################

The :py:mod:`mwave.symbolic` module provides tools for constructing atom interferometers symbolically using SymPy. You define unitary operators (beamsplitters, mirrors, free evolution), compose them into an interferometer sequence, and extract analytic phase expressions. Custom ``Unitary`` subclasses can override ``gen_numeric`` to link symbolic geometry definitions to numerical Bragg pulse simulations.

See the `Interferometer Geometries`_ section for examples of how to use this module.

.. _`Interferometer Geometries`: examples/geometries.ipynb

Interferometer class
====================

.. autoclass:: mwave.symbolic.Interferometer
   :members:

Unitary operators
=================

.. autoclass:: mwave.symbolic.Unitary
   :members:

.. autoclass:: mwave.symbolic.FreeEv
   :members:

.. autoclass:: mwave.symbolic.Beamsplitter
   :members:

.. autoclass:: mwave.symbolic.Mirror
   :members:

Nodes
=====

.. autoclass:: mwave.symbolic.TreeNode
   :members:
   
.. autoclass:: mwave.symbolic.InterferometerNode
   :members:

Helper functions
================

.. autofunction:: mwave.symbolic.set_constants

.. autofunction:: mwave.symbolic.eval_sympy_var