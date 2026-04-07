Integration backends
####################

The :py:func:`mwave.integrate.propagate` function solves the Bloch Hamiltonian for Bragg atom diffraction. Five backends are available, each implementing the same physics but targeting different hardware. The backend can be selected explicitly via the ``backend`` parameter or left as ``None`` for automatic selection.

.. contents:: On this page
   :local:
   :depth: 2

Equations of motion
===================

All backends integrate the same physics: an atom whose momentum is restricted to a discrete ladder of states :math:`|k\rangle` spaced by two photon recoils, coupled by a two-photon drive with time-dependent Rabi frequency :math:`\Omega(t)` and phase :math:`\theta(t)`, and detuned from resonance by :math:`\delta`. Units throughout are :math:`\hbar = 1` with time measured in :math:`1/\omega_\text{r}` (inverse recoil frequency).

Wavefunction evolution (Schrodinger)
------------------------------------

The wavefunction backends (every backend except the density-matrix mode of ``scipy``) integrate

.. math::

   \frac{d|\phi\rangle}{dt}=i\,\frac{\Omega(t)}{2}\left[e^{i(\delta t+\theta(t))}e^{i(-4k-4)t}|k\rangle\langle k+2| + e^{-i(\delta t+\theta(t))}e^{i(4k-4)t}|k\rangle\langle k-2|\right]|\phi\rangle

where :math:`k` indexes momentum states spaced by two photon recoils. The kinematic factors :math:`e^{i(\pm 4k \mp 4)t}` arise from working in the interaction picture relative to the kinetic Hamiltonian.

When the ``scipy`` backend is invoked with ``transformed=True``, the kinetic term is folded back into the propagated state and the equation becomes

.. math::

   \frac{d|\phi\rangle}{dt}=-i\,k^2|k\rangle\langle k|\phi\rangle + i\,\frac{\Omega(t)}{2}\left[e^{i(\delta t+\theta(t))}|k\rangle\langle k+2| + e^{-i(\delta t+\theta(t))}|k\rangle\langle k-2|\right]|\phi\rangle

Density-matrix evolution (Von Neumann)
--------------------------------------

When the ``scipy`` backend is invoked with a non-``None`` ``Gamma_sps``, it instead integrates the Von Neumann equation :math:`d\rho/dt = -i[H,\rho] + \mathcal{L}[\rho]` for the Hamiltonian

.. math::

   H=-\sum_{k}\left[\frac{\Omega(t)}{2}e^{i(\delta t+\theta(t))}e^{i(-4k-4)t}|k\rangle\langle k+2| + \frac{\Omega(t)^*}{2}e^{-i(\delta t+\theta(t))}e^{i(4k-4)t}|k\rangle\langle k-2|\right]

augmented by a phenomenological loss superoperator :math:`\mathcal{L}` that damps the off-diagonal elements of :math:`\rho` at rate :math:`\Gamma_\text{sps}/2` to model single-photon scattering.

Backend selection
=================

When ``backend=None``, ``propagate`` chooses automatically:

- **Single-atom** input (1-D ``phi0``) |rarr| ``'scipy'``
- **Batch** input (2-D ``phi0``) |rarr| ``'numba'``

.. |rarr| unicode:: U+2192

You can override this by passing ``backend='cpp'``, ``backend='gpu'``, etc.

To benchmark all available backends on your machine, call :py:func:`mwave.integrate.score_backends`:

.. code-block:: python

   from mwave.integrate import score_backends
   score_backends()

This runs a standard 5\ :math:`\hbar k` Bragg pulse with 8 atoms through each backend, reports wall-clock time, and flags any that are unavailable due to missing dependencies.

Integration strategies
======================

The backends use two distinct integration strategies:

Adaptive RK45 (scipy, numba)
----------------------------

The ``scipy`` and ``numba`` backends use adaptive step-size control. At each step the integrator estimates the local error and adjusts the step size to keep it within tolerance.

- **scipy** uses SciPy's ``solve_ivp`` with the ``DOP853`` method (8th-order Dormand-Prince). It is the highest-precision option and supports dense (interpolatable) output, the transformed frame, and density-matrix evolution with single-photon scattering (``Gamma_sps``). Single-atom only.

- **numba** uses a custom Dormand-Prince RK45 implementation compiled with Numba. Each step uses the embedded error estimate to adapt ``h``:

  .. code-block:: text

     factor = 0.9 * (tol_step / err) ^ (1/5)
     h_new  = clamp(h * factor, 0.1*h, 10*h)

  where ``tol_step = tol * h / T`` scales the local budget to bound the global error. Atoms are parallelised via Numba ``prange``.

Pilot + fixed-step Richardson (cpp, gpu, metal)
------------------------------------------------

The ``cpp``, ``gpu``, and ``metal`` backends use fixed-step RK4 with Richardson extrapolation to reach the requested tolerance:

1. **Pilot step-size estimate.** A cheap 3-state (``k = [-2, 0, 2]``) adaptive RK45 integration is run via SciPy. The mean accepted step size gives a conservative starting estimate for the fixed-step integrator.

2. **Pre-evaluation.** The time-dependent Rabi envelope :math:`\Omega(t)`, drive phase :math:`e^{i\theta(t)}`, and kinematic coupling terms :math:`e^{i(\pm 4k \mp 4)t}` are evaluated once on a uniform time grid and stored in arrays. This avoids redundant callable evaluations inside the inner loop.

3. **Richardson loop.** Two passes are run at step sizes ``h`` and ``h/2``. The error is estimated as :math:`\max|\phi_{\text{coarse}} - \phi_{\text{fine}}| / 15` (from the RK4 error scaling). If the error exceeds ``tol``, the step count is doubled and the process repeats, up to ``max_halvings`` iterations.

This strategy is well-suited to compiled backends where the inner RK4 loop is extremely fast but cannot easily support embedded error estimates.

Backends
========

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

The default backend for batch simulations. Uses an adaptive Dormand-Prince RK45 integrator compiled with ``@njit(parallel=True, fastmath=True)``.

Each atom is integrated independently with its own per-atom detuning (``delta``) and Rabi scale (``omegas``). The adaptive stepper means no pilot integration or Richardson extrapolation is needed -- a single pass suffices.

This backend operates in float64 throughout. It is the reference implementation against which other backends are validated.

cpp
---

:Dependencies: C++ compiler (``g++`` or ``clang++`` with OpenMP)
:Precision: float32 state, float64 detuning phase accumulation
:Parallelism: OpenMP (``#pragma omp parallel for``)
:Compilation: On first use per unique ``N``; cached to disk at ``src/cpp_build/``

A C++ kernel compiled at runtime via ``g++`` (or Apple Clang + Homebrew libomp on macOS). The compiler is auto-detected from a list of candidates.

The kernel uses fixed-step RK4 with one OpenMP thread per atom. Detuning phases are accumulated in float64 and truncated to float32 at each step to avoid drift, while the wavefunction state is stored in float32 for speed.

Compiled shared libraries are cached on disk (keyed by ``N`` and a source hash), so the ~1-2 second compilation cost is only paid once.

Typical speedup: **2-5x** over the numba backend.

gpu
---

:Dependencies: `CuPy <https://cupy.dev/>`_ and an NVIDIA GPU with CUDA
:Precision: float32 state, float64 detuning phase accumulation (on CPU)
:Parallelism: One CUDA block per atom, ``nextpow2(N)`` threads per block
:Compilation: NVRTC on first use per unique ``N``; cached per session

Two CUDA kernels are compiled at runtime:

1. **Detuning phase precompute** (``bloch_ep_precompute_T``): one thread per atom, computes :math:`e^{i \delta_i t_j}` for all time steps. Float64 accumulation on GPU, stored as float32.

2. **RK4 integration** (``bloch_rk4_noFP64_T``): one block per atom with ``nextpow2(N)`` threads. Uses ping-pong shared memory for the RK4 stages -- each of the four stages reads from one shared-memory buffer and writes to the other, with threadgroup barriers between stages. This avoids register spilling for large ``N``.

The transposed memory layout (atoms as the fast index for detuning phases) ensures coalesced GPU memory access.

Typical speedup: **10-50x** over the numba backend for large ``N`` and many atoms.

metal
-----

:Dependencies: `metalcompute <https://pypi.org/project/metalcompute/>`_ and an Apple Silicon GPU
:Precision: float32 state; detuning phases computed in float64 on CPU
:Parallelism: One threadgroup per atom, ``nextpow2(N)`` threads per group
:Compilation: Metal Shading Language on first use per unique ``N``; cached per session

A Metal compute shader for Apple Silicon. The kernel structure mirrors the CUDA backend (ping-pong shared memory, 4 barriers per RK4 step), but detuning phases :math:`e^{i \delta_i t_j}` are computed on the CPU in float64 and passed to the GPU as float32. This is necessary because Apple Silicon does not natively support float64 in Metal shaders, but produces numerically identical results to the CUDA backend's float64 GPU accumulation.

Dispatch uses ``natoms * nextpow2(N)`` total threads with a fixed threadgroup size of 1024. Since ``nextpow2(N)`` is always a power of two that divides 1024, each atom's threads are guaranteed to land in the same threadgroup.

Typical speedup: **5-20x** over the numba backend.

Precision and numerical consistency
====================================

All backends solve the same Bloch Hamiltonian, but differ in floating-point precision:

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 20

   * - Backend
     - State precision
     - Detuning phase
     - Output
   * - scipy
     - float64
     - float64
     - float64
   * - numba
     - float64
     - float64
     - float64
   * - cpp
     - float32
     - float64 accumulation, float32 per step
     - float64 (upcast)
   * - gpu
     - float32
     - float64 accumulation, float32 storage
     - float64 (upcast)
   * - metal
     - float32
     - float64 on CPU, float32 on GPU
     - float64 (upcast)

The ``cpp``, ``gpu``, and ``metal`` backends return float64 arrays (upcast from float32) for a uniform API. The :py:func:`~mwave.integrate.score_backends` function verifies that all backends agree with the ``numba`` reference to within ``1e-3``.

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
   * - Batch, no GPU available
     - numba
     - No compilation; sufficient for small batches
   * - Batch, macOS with Apple Silicon
     - metal
     - Native GPU; no CUDA dependency
   * - Batch, NVIDIA GPU available
     - gpu
     - Best throughput for large N and many atoms
   * - Batch, Linux/cluster without GPU
     - cpp
     - Good scaling with OpenMP; cached compilation
   * - Very tight tolerance (< 1e-9)
     - scipy
     - Full float64 throughout
