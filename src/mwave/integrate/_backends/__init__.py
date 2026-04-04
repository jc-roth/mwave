"""Backend implementations for :py:func:`mwave.integrate.bloch_rk4`.

Three parallel backends share pre-evaluated envelope/phase/kinematic arrays and
differ only in how the inner RK4 loop is parallelised over atoms:

- ``'python'`` — Numba ``@njit(parallel=True)`` with ``prange``  (no external deps)
- ``'cpp'``    — C++ OpenMP, compiled via g++ on first use, cached per N
- ``'gpu'``    — CUDA via CuPy, compiled via NVRTC on first use, cached per N
"""

import ctypes as _ctypes
import numpy as np
from scipy.integrate import solve_ivp

from ._python import (
    _rk4_bloch_single_kernel,
    _rk4_bloch_batched_kernel,
    _bloch_python_warmup,
    _ensure_bloch_python,
    _rk45_dp_step,
    _rk45_bloch_adaptive,
)
from ._cpp import (
    _CPP_BUILD_DIR,
    _CPP_BLOCH_SOURCE_TMPL,
    _bloch_cpp_libs,
    _compile_bloch_cpp,
    _ensure_bloch_cpp,
)
from ._gpu import (
    _CUDA_BLOCH_SOURCE_TMPL,
    _bloch_gpu_kernels_cache,
    _next_pow2,
    _compile_bloch_gpu,
    _ensure_bloch_gpu,
)
from ._metal import (
    _METAL_BLOCH_SOURCE_TMPL,
    _bloch_metal_cache,
    _compile_bloch_metal,
    _ensure_bloch_metal,
)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _cptr(arr):
    return arr.ctypes.data_as(_ctypes.c_void_p)


def _preeval_rk4_arrays(kvec, t0, nsteps, h, omega, omega_args, phase, phase_args):
    """Pre-evaluate omega, phase, and kinematic arrays on a uniform grid.

    Returns arrays used by :py:func:`_rk4_bloch_single_kernel`:
    ``(env_full, env_half, eph_full, eph_half, kinp_full, kinm_full, kinp_half, kinm_half)``

    :param kvec: Momentum-state grid.
    :param t0: Start time.
    :param nsteps: Number of RK4 steps.
    :param h: Step size.
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments for ``omega``.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments for ``phase``.
    :returns: Tuple of eight pre-evaluated arrays.
    """
    t_full = t0 + np.arange(nsteps + 1, dtype=np.float64) * h
    t_half = t0 + (np.arange(nsteps,     dtype=np.float64) + 0.5) * h

    env_full = np.array([omega(t, omega_args) for t in t_full], dtype=np.float64)
    env_half = np.array([omega(t, omega_args) for t in t_half], dtype=np.float64)
    eph_full = np.exp(1j * np.array([phase(t, phase_args) for t in t_full],
                                     dtype=np.float64)).astype(np.complex128)
    eph_half = np.exp(1j * np.array([phase(t, phase_args) for t in t_half],
                                     dtype=np.float64)).astype(np.complex128)

    base_p = -4.0 * kvec - 4.0
    base_m =  4.0 * kvec - 4.0
    kinp_full = np.ascontiguousarray(np.exp(1j * np.outer(t_full, base_p)))  # (nsteps+1, N)
    kinm_full = np.ascontiguousarray(np.exp(1j * np.outer(t_full, base_m)))
    kinp_half = np.ascontiguousarray(np.exp(1j * np.outer(t_half, base_p)))  # (nsteps, N)
    kinm_half = np.ascontiguousarray(np.exp(1j * np.outer(t_half, base_m)))

    return env_full, env_half, eph_full, eph_half, kinp_full, kinm_full, kinp_half, kinm_half


def _pilot_dt(t0, tfinal, delta, omega, omega_args, phase, phase_args, rtol=1e-8):
    """Determine a step-size suggestion via a cheap pilot RK45 integration.

    Runs ``solve_ivp`` with method ``'RK45'`` and ``rtol=rtol`` on a minimal
    three-state Bragg system ``[-2, 0, 2]``.  Returns the mean step accepted by
    the adaptive solver, which gives a conservative estimate of the step size
    needed by the fixed-step RK4 kernel.

    Uses a plain-Python RHS so that ``omega`` and ``phase`` can be either
    Numba JIT functions or ordinary Python callables.

    :param t0: Start time.
    :param tfinal: End time.
    :param delta: Two-photon detuning.
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments for ``omega``.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments for ``phase``.
    :param rtol: Relative tolerance for the pilot integration.
    :returns: Mean accepted step size as a float.
    """
    kvec3 = np.array([-2.0, 0.0, 2.0])
    phi3  = np.array([0.0, 1.0, 0.0], dtype=np.complex128)

    def _rhs3(t, phi):
        phi_p1 = np.zeros(3, dtype=np.complex128)
        phi_p1[:-1] = phi[1:]
        phi_m1 = np.zeros(3, dtype=np.complex128)
        phi_m1[1:]  = phi[:-1]
        oval     = float(omega(t, omega_args))
        phaseval = float(phase(t, phase_args))
        ep  = np.exp( 1j * (delta * t + phaseval))
        epc = np.exp(-1j * (delta * t + phaseval))
        return (1j * oval / 2.0 *
                (ep  * np.exp(1j * (-4.0 * kvec3 - 4.0) * t) * phi_p1 +
                 epc * np.exp(1j * ( 4.0 * kvec3 - 4.0) * t) * phi_m1))

    sol = solve_ivp(
        _rhs3, [t0, tfinal], phi3,
        method='RK45', rtol=rtol, atol=rtol * 1e-3, dense_output=False,
    )
    # Use the mean adaptive step size rather than the minimum.  The minimum
    # can be orders-of-magnitude smaller than the typical step (e.g. when the
    # solver briefly undershoots near a steep derivative) and leads to a
    # catastrophically large initial nsteps.  Richardson extrapolation handles
    # accuracy regardless of the starting nsteps estimate.
    n_steps = max(len(sol.t) - 1, 1)
    return float((tfinal - t0) / n_steps)


def _run_batched(backend, phi0, deltas, omegas, h, t0,
                 env_full, env_half, eph_full, eph_half,
                 kinp_full, kinm_full, kinp_half, kinm_half):
    """Run one fixed-step RK4 pass over all atoms with the chosen backend.

    :returns: ``(natoms, N) complex128`` wavefunction after integration.
    """
    natoms, N = phi0.shape
    nsteps    = env_half.shape[0]

    if backend == 'python':
        _ensure_bloch_python()
        phi_all           = phi0.copy()
        delta_phase_inits = deltas * t0
        _rk4_bloch_batched_kernel(
            phi_all, omegas, h,
            env_full, env_half, eph_full, eph_half,
            kinp_full, kinm_full, kinp_half, kinm_half,
            deltas, delta_phase_inits)
        return phi_all

    elif backend == 'cpp':
        lib   = _ensure_bloch_cpp(N)
        phi32 = np.ascontiguousarray(phi0,      dtype=np.complex64)
        ef32  = np.ascontiguousarray(env_full,  dtype=np.float32)
        eh32  = np.ascontiguousarray(env_half,  dtype=np.float32)
        eph_f = np.ascontiguousarray(eph_full,  dtype=np.complex64)
        eph_h = np.ascontiguousarray(eph_half,  dtype=np.complex64)
        kpf32 = np.ascontiguousarray(kinp_full, dtype=np.complex64)
        kmf32 = np.ascontiguousarray(kinm_full, dtype=np.complex64)
        kph32 = np.ascontiguousarray(kinp_half, dtype=np.complex64)
        kmh32 = np.ascontiguousarray(kinm_half, dtype=np.complex64)
        om32  = np.ascontiguousarray(omegas,    dtype=np.float32)
        dl64  = np.ascontiguousarray(deltas,    dtype=np.float64)
        lib.rk4_bloch_f32(
            _cptr(phi32), _ctypes.c_int(natoms), _ctypes.c_float(float(h)),
            _ctypes.c_double(float(t0)),
            _cptr(ef32), _cptr(eh32), _cptr(eph_f), _cptr(eph_h),
            _ctypes.c_int(nsteps), _cptr(om32), _cptr(dl64),
            _cptr(kpf32), _cptr(kmf32), _cptr(kph32), _cptr(kmh32),
        )
        return phi32.astype(np.complex128)

    elif backend == 'gpu':
        import cupy as cp
        k_ep, k_rk4 = _ensure_bloch_gpu(N)
        blk_ep  = 128
        blk_rk4 = _next_pow2(N)

        phi_gpu   = cp.asarray(phi0.astype(np.complex64))
        ef_gpu    = cp.asarray(env_full.astype(np.float32))
        eh_gpu    = cp.asarray(env_half.astype(np.float32))
        eph_f_gpu = cp.asarray(eph_full.astype(np.complex64))
        eph_h_gpu = cp.asarray(eph_half.astype(np.complex64))
        kpf_gpu   = cp.asarray(kinp_full.astype(np.complex64))
        kmf_gpu   = cp.asarray(kinm_full.astype(np.complex64))
        kph_gpu   = cp.asarray(kinp_half.astype(np.complex64))
        kmh_gpu   = cp.asarray(kinm_half.astype(np.complex64))
        dl_gpu    = cp.asarray(deltas.astype(np.float64))
        om_gpu    = cp.asarray(omegas.astype(np.float32))

        ep_full_T_gpu = cp.empty((nsteps + 1) * natoms, dtype=cp.complex64)
        ep_half_T_gpu = cp.empty(nsteps * natoms,        dtype=cp.complex64)

        h_f32  = np.float32(h)
        h_f64  = np.float64(h)
        t0_f64 = np.float64(t0)

        k_ep(
            ((natoms + blk_ep - 1) // blk_ep,), (blk_ep,),
            (ep_full_T_gpu, ep_half_T_gpu, dl_gpu,
             natoms, nsteps, h_f64, t0_f64),
        )
        k_rk4(
            (natoms,), (blk_rk4,),
            (phi_gpu, natoms, h_f32, nsteps,
             ef_gpu, eh_gpu, om_gpu,
             ep_full_T_gpu, ep_half_T_gpu,
             eph_f_gpu, eph_h_gpu,
             kpf_gpu, kmf_gpu, kph_gpu, kmh_gpu),
        )
        cp.cuda.Device(0).synchronize()
        return phi_gpu.get().astype(np.complex128)

    elif backend == 'metal':
        dev, fn = _ensure_bloch_metal(N)
        blk = _next_pow2(N)

        # Detuning phase: computed on CPU in float64, stored as float32 complex.
        # This matches the precision of the CUDA bloch_ep_precompute_T kernel
        # (float64 accumulation → float32 output) without requiring float64 in
        # Metal shaders (which Apple Silicon does not support natively).
        t_full = t0 + np.arange(nsteps + 1, dtype=np.float64) * h
        t_half = t0 + (np.arange(nsteps,     dtype=np.float64) + 0.5) * h
        ep_full_T = np.ascontiguousarray(
            np.exp(1j * np.outer(t_full, deltas)).astype(np.complex64))  # (nsteps+1, natoms)
        ep_half_T = np.ascontiguousarray(
            np.exp(1j * np.outer(t_half, deltas)).astype(np.complex64))  # (nsteps,   natoms)

        # Convert shared arrays to float32
        ef32  = np.ascontiguousarray(env_full,  dtype=np.float32)
        eh32  = np.ascontiguousarray(env_half,  dtype=np.float32)
        om32  = np.ascontiguousarray(omegas,    dtype=np.float32)
        eph_f = np.ascontiguousarray(eph_full.astype(np.complex64))
        eph_h = np.ascontiguousarray(eph_half.astype(np.complex64))
        kpf32 = np.ascontiguousarray(kinp_full.astype(np.complex64))
        kmf32 = np.ascontiguousarray(kinm_full.astype(np.complex64))
        kph32 = np.ascontiguousarray(kinp_half.astype(np.complex64))
        kmh32 = np.ascontiguousarray(kinm_half.astype(np.complex64))

        # Pack h as bits in params[2] to avoid metalcompute's 1-element scalar
        # heuristic (which would cause a DeprecationWarning for ndim>0→scalar).
        params = np.array([natoms, nsteps,
                           np.float32(h).view(np.int32)], dtype=np.int32)

        # phi_all is in/out — must use a dev.buffer so Metal can write back.
        phi32   = np.ascontiguousarray(phi0, dtype=np.complex64)
        phi_buf = dev.buffer(phi32.nbytes)
        memoryview(phi_buf)[:phi32.nbytes] = phi32.tobytes()

        # Dispatch: natoms * blk total threads.
        # metalcompute uses a fixed threadgroup size of 1024; since blk
        # (a power-of-two ≤ 1024) divides 1024, every atom's threads land
        # in the same threadgroup and the threadgroup scratch is coherent.
        fn(int(natoms) * blk,
           phi_buf, params,
           ef32, eh32, om32,
           ep_full_T, ep_half_T,
           eph_f, eph_h,
           kpf32, kmf32, kph32, kmh32)

        result = np.frombuffer(phi_buf, dtype=np.complex64).reshape(natoms, N).copy()
        return result.astype(np.complex128)

    else:
        raise ValueError(
            f"bloch_rk4: unknown backend {backend!r}; "
            "choose 'python', 'cpp', 'gpu', or 'metal'")
