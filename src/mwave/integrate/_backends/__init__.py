"""Backend implementations for :py:func:`mwave.integrate.propagate`.

- ``'scipy'``  — adaptive ``solve_ivp`` (single-atom only, supports dense output)
- ``'numba'``  — Numba ``@njit(parallel=True)`` with ``prange``  (no external deps)
- ``'cpp'``    — C++ OpenMP, compiled via g++ on first use, cached per N
- ``'gpu'``    — CUDA via CuPy, compiled via NVRTC on first use, cached per N
- ``'metal'``  — Metal compute, compiled via metalcompute on first use, cached per N
"""

import ctypes as _ctypes
import numpy as np

from ._scipy import _run_scipy

from ._numba import (
    _rk45_dp_step,
    _rk45_bloch_adaptive,
)
from ._cpp import _ensure_bloch_cpp
from ._gpu import _next_pow2, _ensure_bloch_gpu
from ._metal import _ensure_bloch_metal


# ── Shared utilities ──────────────────────────────────────────────────────────

def _cptr(arr):
    return arr.ctypes.data_as(_ctypes.c_void_p)


def _preeval_rk4_arrays(kvec, t0, nsteps, h, omega, omega_args, phase, phase_args):
    """Pre-evaluate omega, phase, and kinematic arrays on a uniform grid.

    Returns the arrays consumed by the fixed-step RK4 kernels in
    :py:func:`_run_batched` (cpp/gpu/metal backends):
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


def _run_batched(backend, phi0, deltas, omegas, h, t0,
                 env_full, env_half, eph_full, eph_half,
                 kinp_full, kinm_full, kinp_half, kinm_half):
    """Run one fixed-step RK4 pass over all atoms with the chosen backend.

    Used by the cpp/gpu/metal pilot+Richardson path in
    :py:func:`mwave.integrate.propagate`.  The numba backend uses
    :py:func:`_rk45_bloch_adaptive` directly and does not go through here.

    :returns: ``(natoms, N) complex128`` wavefunction after integration.
    """
    natoms, N = phi0.shape
    nsteps    = env_half.shape[0]

    if backend == 'cpp':
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
            f"_run_batched: unknown backend {backend!r}; "
            "choose 'cpp', 'gpu', or 'metal'")
