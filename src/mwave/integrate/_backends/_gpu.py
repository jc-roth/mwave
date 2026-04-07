"""CUDA/GPU backend for bloch_rk4.

Public entry point: :py:func:`_ensure_bloch_gpu`, which compiles the CUDA
source via NVRTC on first use (cached per ``N``) and returns the
``(bloch_ep_precompute_T, bloch_rk4_noFP64_T)`` raw kernels.  Also exports
:py:func:`_next_pow2`, the threadblock-size helper used by both the GPU
and Metal backends.
"""

import ctypes as _ctypes
import os as _os


_CUDA_BLOCH_SOURCE_TMPL = r"""
#include <cuComplex.h>
#define NN PLACEHOLDER_NN
#define BLK_RK4 PLACEHOLDER_BLK_RK4

/* One thread per atom. Accumulates exp(i*delta_i*t_j) starting from t0_d.
 * Transposed layout [j*natoms+atom] for coalesced writes. */
extern "C" __global__ void bloch_ep_precompute_T(
    cuFloatComplex* __restrict__ ep_full_T,
    cuFloatComplex* __restrict__ ep_half_T,
    const double*   __restrict__ deltas_d,
    int natoms, int nsteps, double h_d, double t0_d)
{
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= natoms) return;

    double delta   = deltas_d[atom];
    double ep_re   = cos(delta * t0_d), ep_im   = sin(delta * t0_d);
    double ci_re   = cos(delta * h_d),  ci_im   = sin(delta * h_d);
    double ci_h_re = cos(delta * h_d * 0.5), ci_h_im = sin(delta * h_d * 0.5);

    ep_full_T[0 * natoms + atom] = make_cuFloatComplex((float)ep_re, (float)ep_im);

    for (int j = 0; j < nsteps; j++) {
        double eph_re = ep_re*ci_h_re - ep_im*ci_h_im;
        double eph_im = ep_re*ci_h_im + ep_im*ci_h_re;
        ep_half_T[j * natoms + atom] = make_cuFloatComplex((float)eph_re, (float)eph_im);

        double ep1_re = ep_re*ci_re - ep_im*ci_im;
        double ep1_im = ep_re*ci_im + ep_im*ci_re;
        ep_full_T[(j + 1) * natoms + atom] = make_cuFloatComplex((float)ep1_re, (float)ep1_im);

        ep_re = ep1_re; ep_im = ep1_im;
    }
}

/* One block per atom (BLK_RK4 threads). Ping-pong shared memory, 4 syncs/step.
 * ep_full_T/ep_half_T: detuning phase only (transposed, broadcast within block).
 * eph_full/eph_half:   arbitrary phase(t), shared across atoms (not transposed).
 * Combined coupling phase = ep_delta * eph(t_j). */
extern "C" __global__ void bloch_rk4_noFP64_T(
    cuFloatComplex* __restrict__ phi_all,
    int natoms, float h, int nsteps,
    const float*          __restrict__ env_full,
    const float*          __restrict__ env_half,
    const float*          __restrict__ omegas,
    const cuFloatComplex* __restrict__ ep_full_T,
    const cuFloatComplex* __restrict__ ep_half_T,
    const cuFloatComplex* __restrict__ eph_full,
    const cuFloatComplex* __restrict__ eph_half,
    const cuFloatComplex* __restrict__ kinp_full,
    const cuFloatComplex* __restrict__ kinm_full,
    const cuFloatComplex* __restrict__ kinp_half,
    const cuFloatComplex* __restrict__ kinm_half)
{
    int atom = blockIdx.x;
    int n    = threadIdx.x;

    __shared__ float arr_a_r[BLK_RK4], arr_a_i[BLK_RK4];
    __shared__ float arr_b_r[BLK_RK4], arr_b_i[BLK_RK4];

    float phi_r = 0.0f, phi_i = 0.0f;
    if (n < NN) {
        cuFloatComplex v = phi_all[atom * NN + n];
        phi_r = cuCrealf(v); phi_i = cuCimagf(v);
    }
    arr_a_r[n] = phi_r; arr_a_i[n] = phi_i;
    __syncthreads();

    float omega = omegas[atom];
    float h2 = 0.5f * h;
    float h6 = h / 6.0f;

    for (int j = 0; j < nsteps; j++) {
        /* Detuning phase (transposed table, broadcast: same atom for all threads) */
        cuFloatComplex epd0v = ep_full_T[ j      * natoms + atom];
        cuFloatComplex epdhv = ep_half_T[ j      * natoms + atom];
        cuFloatComplex epd1v = ep_full_T[(j + 1) * natoms + atom];

        float epd0_re = cuCrealf(epd0v), epd0_im = cuCimagf(epd0v);
        float epd_h_re = cuCrealf(epdhv), epd_h_im = cuCimagf(epdhv);
        float epd1_re = cuCrealf(epd1v), epd1_im = cuCimagf(epd1v);

        /* Shared phase (not transposed, indexed by j only) */
        cuFloatComplex eph0v  = eph_full[j];
        cuFloatComplex ephh_v = eph_half[j];
        cuFloatComplex eph1v  = eph_full[j + 1];

        float eph0_re = cuCrealf(eph0v),  eph0_im = cuCimagf(eph0v);
        float eph_h_re = cuCrealf(ephh_v), eph_h_im = cuCimagf(ephh_v);
        float eph1_re = cuCrealf(eph1v),  eph1_im = cuCimagf(eph1v);

        /* Combined coupling phase: ep_total = ep_delta * eph(t) */
        float ep0_re  = epd0_re*eph0_re  - epd0_im*eph0_im;
        float ep0_im  = epd0_re*eph0_im  + epd0_im*eph0_re;
        float ep1h_re = epd_h_re*eph_h_re - epd_h_im*eph_h_im;
        float ep1h_im = epd_h_re*eph_h_im + epd_h_im*eph_h_re;
        float ep1_re  = epd1_re*eph1_re  - epd1_im*eph1_im;
        float ep1_im  = epd1_re*eph1_im  + epd1_im*eph1_re;

        float o0 = env_full[j], o1 = env_half[j], o2 = env_full[j + 1];

        float a0 = omega * o0 * 0.5f;
        float c0p_re = -a0*ep0_im,   c0p_im = a0*ep0_re;
        float c0m_re =  a0*ep0_im,   c0m_im = a0*ep0_re;

        float a1 = omega * o1 * 0.5f;
        float c1p_re = -a1*ep1h_im, c1p_im = a1*ep1h_re;
        float c1m_re =  a1*ep1h_im, c1m_im = a1*ep1h_re;

        float a2 = omega * o2 * 0.5f;
        float c2p_re = -a2*ep1_im,  c2p_im = a2*ep1_re;
        float c2m_re =  a2*ep1_im,  c2m_im = a2*ep1_re;

        /* ── k1: reads arr_a, writes arr_b = phi + h/2*k1 ─────────────────── */
        float k1r = 0.0f, k1i = 0.0f;
        if (n < NN) {
            cuFloatComplex kpv = kinp_full[j * NN + n];
            cuFloatComplex kmv = kinm_full[j * NN + n];
            float kp_re = cuCrealf(kpv), kp_im = cuCimagf(kpv);
            float km_re = cuCrealf(kmv), km_im = cuCimagf(kmv);
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c0p_re*kp_re - c0p_im*kp_im;
                float t_im = c0p_re*kp_im + c0p_im*kp_re;
                s_re += t_re*arr_a_r[n+1] - t_im*arr_a_i[n+1];
                s_im += t_re*arr_a_i[n+1] + t_im*arr_a_r[n+1];
            }
            if (n > 0) {
                float t_re = c0m_re*km_re - c0m_im*km_im;
                float t_im = c0m_re*km_im + c0m_im*km_re;
                s_re += t_re*arr_a_r[n-1] - t_im*arr_a_i[n-1];
                s_im += t_re*arr_a_i[n-1] + t_im*arr_a_r[n-1];
            }
            k1r = s_re; k1i = s_im;
            arr_b_r[n] = phi_r + h2*k1r;
            arr_b_i[n] = phi_i + h2*k1i;
        } else {
            arr_b_r[n] = 0.0f; arr_b_i[n] = 0.0f;
        }
        __syncthreads();  /* #1 */

        /* ── k2: reads arr_b, writes arr_a = phi + h/2*k2 ─────────────────── */
        float k2r = 0.0f, k2i = 0.0f;
        if (n < NN) {
            cuFloatComplex kpv = kinp_half[j * NN + n];
            cuFloatComplex kmv = kinm_half[j * NN + n];
            float kp_re = cuCrealf(kpv), kp_im = cuCimagf(kpv);
            float km_re = cuCrealf(kmv), km_im = cuCimagf(kmv);
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c1p_re*kp_re - c1p_im*kp_im;
                float t_im = c1p_re*kp_im + c1p_im*kp_re;
                s_re += t_re*arr_b_r[n+1] - t_im*arr_b_i[n+1];
                s_im += t_re*arr_b_i[n+1] + t_im*arr_b_r[n+1];
            }
            if (n > 0) {
                float t_re = c1m_re*km_re - c1m_im*km_im;
                float t_im = c1m_re*km_im + c1m_im*km_re;
                s_re += t_re*arr_b_r[n-1] - t_im*arr_b_i[n-1];
                s_im += t_re*arr_b_i[n-1] + t_im*arr_b_r[n-1];
            }
            k2r = s_re; k2i = s_im;
            arr_a_r[n] = phi_r + h2*k2r;
            arr_a_i[n] = phi_i + h2*k2i;
        }
        __syncthreads();  /* #2 */

        /* ── k3: reads arr_a, writes arr_b = phi + h*k3 ───────────────────── */
        float k3r = 0.0f, k3i = 0.0f;
        if (n < NN) {
            cuFloatComplex kpv = kinp_half[j * NN + n];
            cuFloatComplex kmv = kinm_half[j * NN + n];
            float kp_re = cuCrealf(kpv), kp_im = cuCimagf(kpv);
            float km_re = cuCrealf(kmv), km_im = cuCimagf(kmv);
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c1p_re*kp_re - c1p_im*kp_im;
                float t_im = c1p_re*kp_im + c1p_im*kp_re;
                s_re += t_re*arr_a_r[n+1] - t_im*arr_a_i[n+1];
                s_im += t_re*arr_a_i[n+1] + t_im*arr_a_r[n+1];
            }
            if (n > 0) {
                float t_re = c1m_re*km_re - c1m_im*km_im;
                float t_im = c1m_re*km_im + c1m_im*km_re;
                s_re += t_re*arr_a_r[n-1] - t_im*arr_a_i[n-1];
                s_im += t_re*arr_a_i[n-1] + t_im*arr_a_r[n-1];
            }
            k3r = s_re; k3i = s_im;
            arr_b_r[n] = phi_r + h*k3r;
            arr_b_i[n] = phi_i + h*k3i;
        }
        __syncthreads();  /* #3 */

        /* ── k4 + phi update: reads arr_b ──────────────────────────────────── */
        if (n < NN) {
            cuFloatComplex kpv = kinp_full[(j + 1) * NN + n];
            cuFloatComplex kmv = kinm_full[(j + 1) * NN + n];
            float kp_re = cuCrealf(kpv), kp_im = cuCimagf(kpv);
            float km_re = cuCrealf(kmv), km_im = cuCimagf(kmv);
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c2p_re*kp_re - c2p_im*kp_im;
                float t_im = c2p_re*kp_im + c2p_im*kp_re;
                s_re += t_re*arr_b_r[n+1] - t_im*arr_b_i[n+1];
                s_im += t_re*arr_b_i[n+1] + t_im*arr_b_r[n+1];
            }
            if (n > 0) {
                float t_re = c2m_re*km_re - c2m_im*km_im;
                float t_im = c2m_re*km_im + c2m_im*km_re;
                s_re += t_re*arr_b_r[n-1] - t_im*arr_b_i[n-1];
                s_im += t_re*arr_b_i[n-1] + t_im*arr_b_r[n-1];
            }
            phi_r += h6*(k1r + 2.0f*k2r + 2.0f*k3r + s_re);
            phi_i += h6*(k1i + 2.0f*k2i + 2.0f*k3i + s_im);
            arr_a_r[n] = phi_r;
            arr_a_i[n] = phi_i;
        }
        __syncthreads();  /* #4 */
    }

    if (n < NN) {
        phi_all[atom * NN + n] = make_cuFloatComplex(phi_r, phi_i);
    }
}
"""

_bloch_gpu_kernels_cache = {}   # N → (kernel_ep, kernel_rk4)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _compile_bloch_gpu(N):
    blk    = _next_pow2(N)
    source = (_CUDA_BLOCH_SOURCE_TMPL
              .replace('PLACEHOLDER_NN',      str(N))
              .replace('PLACEHOLDER_BLK_RK4', str(blk)))

    # Optional NVRTC / NVCC path setup (mirrors rk4_gpu_v2)
    try:
        _nvrtc = _os.path.join(_os.path.expanduser('~'), '.local', 'lib',
                               'python3.12', 'site-packages', 'nvidia',
                               'cuda_nvrtc', 'lib', 'libnvrtc.so.12')
        if _os.path.exists(_nvrtc):
            _ctypes.CDLL(_nvrtc)
        _nvcc = _os.path.join(_os.path.expanduser('~'), '.local', 'lib',
                              'python3.12', 'site-packages', 'nvidia', 'cuda_nvcc')
        if _os.path.isdir(_nvcc) and 'CUDA_PATH' not in _os.environ:
            _os.environ['CUDA_PATH'] = _nvcc
    except Exception:
        pass

    import cupy as cp
    opts  = ('--std=c++14', '--use_fast_math')
    k_ep  = cp.RawKernel(source, 'bloch_ep_precompute_T', options=opts)
    k_rk4 = cp.RawKernel(source, 'bloch_rk4_noFP64_T',   options=opts)
    return k_ep, k_rk4


def _ensure_bloch_gpu(N):
    if N not in _bloch_gpu_kernels_cache:
        _bloch_gpu_kernels_cache[N] = _compile_bloch_gpu(N)
    return _bloch_gpu_kernels_cache[N]
