Integration and utilities reference
###################################

This section documents the lower-level modules that power numerical simulations: integration of the Bloch Hamiltonian, helper functions for setting up simulation initial conditions, and utilities for precomputing Bragg pulse lookup tables.

Integrate module
================

The :py:mod:`mwave.integrate` module contains functions for integrating the Bloch Hamiltonian to simulate Bragg diffraction pulses. Most functions use :py:mod:`numba` for performance.

.. automodule:: mwave.integrate
   :members:

Simulation utils module
=======================

The :py:mod:`mwave.simulation_utils` module provides helpers for initializing atom clouds and computing interferometer observables.

.. automodule:: mwave.simulation_utils
   :members:
