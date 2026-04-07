Integration backends
####################

The :py:func:`mwave.integrate.propagate` function integrates the Hamiltonian for Bragg diffraction and Bloch oscillations. Two backends are available. The backend can be selected explicitly via the ``backend`` parameter or left as ``None`` for default selection.

.. contents:: On this page
   :local:
   :depth: 2

Equations of motion
===================

The equations of motion are derived from the Schrodinger equation for the Hamiltonian describing Bragg diffraction and Bloch oscillations, which yields a system of ODEs:

.. math::

   \frac{d|\phi\rangle}{dt}=i\,\frac{\Omega(t)}{2}\left[e^{i(\delta t+\theta(t))}e^{i(-4k-4)t}|k\rangle\langle k+2| + e^{-i(\delta t+\theta(t))}e^{i(4k-4)t}|k\rangle\langle k-2|\right]|\phi\rangle

where :math:`k` indexes momentum states spaced by two photon recoils, :math:`\Omega(t)` is the time-dependent effective Rabi frequency, :math:`\theta(t)` is the time-dependent phase, and :math:`\delta` is the two-photon detuning. The :py:func:`mwave.integrate.propagate` function integrates this system of ODEs using a different method depending on the selected backend. These different methods are detailed below.

The :py:func:`mwave.integrate.propagate` function accepts an argument :code:`transformed`. If :code:`transformed=True` then the right hand side is evaluated in the following frame:

.. math::

   \frac{d|\phi\rangle}{dt}=-i\,k^2|k\rangle\langle k|\phi\rangle + i\,\frac{\Omega(t)}{2}\left[e^{i(\delta t+\theta(t))}|k\rangle\langle k+2| + e^{-i(\delta t+\theta(t))}|k\rangle\langle k-2|\right]|\phi\rangle

Density-matrix evolution
------------------------

When the ``scipy`` backend is invoked with ``Gamma_sps``, it integrates the Von Neumann evolution equation (i.e. :math:`[H,\rho]`) where

.. math::

    H=-\hbar\sum_{k}\left[\frac{\Omega_\text{eff}(t)}{2}e^{i(\delta t+\theta(t))}e^{i\omega_\text{r}(-4k-4)}|k\rangle\langle k+2|+\frac{\Omega_\text{eff}(t)^*}{2}e^{-i(\delta t+\theta(t))}e^{i\omega_\text{r}(4k-4)}|k\rangle\langle k-2|\right]

where :math:`\hbar=1` and the sum over :math:`k` is limited to the values of :math:`k` defined by :code:`hkvec` and :code:`vkvec`.

The parameter :code:`rho` is supplied as a vector (this makes it compatible with :code:`scipy.integrate.solve_ivp`). This is then converted to a matrix via :code:`np.reshape(rho, (len(kvec), len(kvec)))` internally. The matrices :code:`hkvec` and :code:`vkvec` are composed of horizontal or vertical vectors of the momentum state grid stacked together.

The parameter :code:`loss_mat` is the loss matrix.

Backend selection
=================

When ``backend=None``, ``propagate`` selects backends based on the shape of ``phi0``:

- **Single-atom** input (1-D ``phi0``) |rarr| ``'scipy'``
- **Batch** input (2-D ``phi0``) |rarr| ``'numba'``

.. |rarr| unicode:: U+2192

You can override this by passing ``backend='scipy'`` or ``backend='numba'`` explicitly.

To benchmark the available backends on your machine, call :py:func:`mwave.integrate.score_backends`:

.. code-block:: python

   from mwave.integrate import score_backends
   score_backends()

This runs a standard 5\ :math:`\hbar k` Bragg pulse through each backend for a single atom and then a batch of atoms, reporting the wall-clock time. The number of atoms used in the batch can be adjusted via ``natoms=10000`` or similar.

.. note::

  To score from the command line run ``python -c "from mwave.integrate import score_backends; score_backends()"``

Integration strategies
======================

Both backends use adaptive step-size Runge-Kutta integration. At each step the integrator estimates the local error and adjusts the step size to keep it within tolerance.

- **scipy** uses SciPy's ``solve_ivp`` with the ``DOP853`` method (8th-order Dormand-Prince). It is the highest-precision option and supports dense (interpolatable) output, the transformed frame, and density-matrix evolution with single-photon scattering (``Gamma_sps``). Single-atom only.

- **numba** uses a custom Dormand-Prince RK45 implementation compiled with Numba. Each step uses the embedded error estimate to adapt ``h``:

  .. code-block:: text

     factor = 0.9 * (tol_step / err) ^ (1/5)
     h_new  = clamp(h * factor, 0.1*h, 10*h)

  where ``tol_step = tol * h / T`` scales the local budget to bound the global error. The local error is taken as the maximum across all atoms and momentum states, so a single step size is chosen that satisfies the worst-case atom. The per-atom RK45 stages themselves are evaluated in parallel via Numba ``prange``.

Backends
========

In the descriptions below, ``N`` denotes the number of momentum states in the simulation, i.e. ``len(kvec)``, and ``natoms`` denotes the number of atoms in a batch.

scipy
-----

:Dependencies: None (uses SciPy, which is a core dependency)
:Precision: float64
:Parallelism: None (single-atom only)
:Compilation: None

The scipy backend delegates to :func:`scipy.integrate.solve_ivp`. It is the only backend that supports:

- **Dense output** (``dense=True``): returns an interpolatable solution accessible via ``result.scipy_sol.sol``.
- **Transformed frame** (``transformed=True``): includes the :math:`-ik^2|k\rangle\langle k|` kinetic term in the Hamiltonian.
- **Density-matrix evolution** (``Gamma_sps``): integrates the Von Neumann equation with a loss matrix to model single-photon scattering.

The ``result.scipy_sol`` attribute gives access to the full :class:`~scipy.integrate.OdeResult`, and ``result.plot()`` produces a three-panel diagnostic figure (populations, Rabi frequency, and phase vs. time).

Use this backend when you need high precision, diagnostic plots, or any of the features above.

numba
-----

:Dependencies: Numba
:Precision: float64
:Parallelism: Numba ``prange`` over atoms (multi-core CPU)
:Compilation: JIT on first call (~10 ms warm-up)

The default backend for batch simulations. Uses a custom adaptive Dormand-Prince RK45 integrator (DOP5(4) embedded pair) compiled with ``@njit(parallel=True)``.

**Algorithm.** All atoms in a batch are advanced in together: they share a single time grid and a single adaptive step size. At each step:

1. The shared time-dependent driving Rabi envelope :math:`\Omega(t)`, laser phase :math:`e^{i\theta(t)}`, and momentum-grid phase factors :math:`e^{i(\pm 4k\mp 4)t}` — is pre-evaluated **once** at the six unique RK45 stage times.

2. The kernel computes all six RK45 stages, the 5th-order solution, and the embedded error estimate for every atom in parallel via ``prange``. This is possible because the evolution of each atom is independent from every other atom.

3. The step error is the max-norm of ``phi_err`` over all atoms and momentum states. Step size is adapted by a standard 5th-order I-controller (``factor = 0.9 * (tol_step / err)^(1/5)``) with ``tol_step = tol * h / T`` so that the **global** error stays bounded by ``tol``.

The shared driving evaluation and the JIT-compiled inner loop together eliminate essentially all per-step Python overhead.

This approach was discovered through an LLM-in-a-loop optimisation process inspired by Google DeepMind's AlphaEvolve. The discovered approach appears to be the same as the one proposed in Utkarsh et al., `arXiv:2304.06835 <https://arxiv.org/abs/2304.06835>`_ (and implemented in Julia's `DiffEqGPU.jl <https://github.com/SciML/DiffEqGPU.jl>`_ as ``EnsembleGPUKernel``), and so should not be considered novel.

This backend operates in float64 throughout.

Choosing a backend
==================

.. list-table::
   :header-rows: 1
   :widths: 30 15 35

   * - Scenario
     - Backend
     - Rationale
   * - Single atom, need dense output or plots
     - scipy
     - Only backend with interpolatable output
   * - Single atom, default
     - scipy
     - Auto-selected; highest precision
   * - Batch
     - numba
     - Auto-selected; parallelised over atoms
   * - Very tight tolerance (< 1e-9)
     - scipy
     - DOP853 is higher order than RK45
