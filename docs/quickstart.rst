Quickstart
##########

:code:`mwave` is a python library designed to help explore new matterwave interferometer geometries. :code:`mwave` also provides functions to numerically solve the Bloch Hamiltonian that describes Bragg diffraction and Bloch oscillations.

Installing :code:`mwave`
========================

Stable releases of :code:`mwave` can be downloaded from `Github`_.

.. _Github: https://github.com/jc-roth/mwave/releases

The latest version of :code:`mwave` can be installed directly from Github via

.. code-block::

   pip install git+https://github.com/jc-roth/mwave.git

An example
==========

:code:`mwave` is losely broken up into two parts: a module for symbolically defining arbitrary interferometer geometries, and a module for numerically calculating Bragg and Bloch processes. Let's first try defining a Mach-Zender interferometer geometry. Start by defining

*insert example*

See the :ref:`geometries_example` section for more in-depth examples.

Next we can use the :py:meth:`mwave.integrate.gbragg` function to integrate some initial momentum state through a Bragg diffraction beamsplitter

*insert example*

Lets say that we want to study the systematics introduced by the Bragg diffraction process in our Mach-Zender geometry. To do this we need to combine the numerical computation we've made using :py:meth:`mwave.integrate.gbragg` with our symbolic representation of the interferometer geometry. This is accomplished in a straightforward way by defining a custom :py:class:`mwave.symbolic.Unitary` class.

*insert example*

As the user becomes more and more interested in a specific geometry it becomes more and more non-trivial to incorporate new systematic effects using this method. To help :code:`mwave` can generate code specifically for generating a particular interferometer geometry.

*insert example*

Future changes
==============

Several changes are planned for the future:

- Support installation from Pypi
- Adding a variable in :py:class:`mwave.symbolic.InterferometerNode` to track the internal state.