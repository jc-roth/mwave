"""Metal/Apple Silicon backend for bloch_rk4.

Contains:
- Metal MSL source template                   (_METAL_BLOCH_SOURCE_TMPL)
- Compiled kernel cache                       (_bloch_metal_cache)
- Compile-on-first-use helper                 (_compile_bloch_metal)
- Cache-aware ensure helper                   (_ensure_bloch_metal)

Design notes
------------
metalcompute v0.2.x dispatches all threads in threadgroups of exactly 1024.
The kernel uses ``thread_position_in_grid`` (``gid``) and
``thread_position_in_threadgroup`` (``tid``) to derive:

  atom = gid / BLK_RK4   (global atom index)
  n    = tid % BLK_RK4   (= gid % BLK_RK4, valid because BLK_RK4 | 1024)

Threadgroup scratch arrays of 1024 elements are declared inline in the
kernel and indexed by ``tid``.  Because BLK_RK4 (a power-of-two ≤ 1024)
evenly divides the 1024-element threadgroup, every atom's BLK_RK4 threads
land in the same threadgroup and their scratch slots are contiguous and
non-overlapping.  Neighbor accesses ``arr[tid±1]`` are guarded by the
``n < NN-1`` / ``n > 0`` conditions inherited from the CUDA kernel, so no
atom crosses its neighbour's scratch region.

Apple Silicon does not support float64 in Metal shaders.  The detuning-phase
arrays (``ep_full_T``, ``ep_half_T``) are therefore computed on the CPU in
float64 and converted to complex64 before being uploaded to the GPU, exactly
matching the precision of the CUDA backend's ``bloch_ep_precompute_T``
kernel (which also produces complex64 output).
"""

import numpy as _np

from ._gpu import _next_pow2


# ── MSL source template ───────────────────────────────────────────────────────

_METAL_BLOCH_SOURCE_TMPL = r"""
#include <metal_stdlib>
using namespace metal;

#define NN       PLACEHOLDER_NN
#define BLK_RK4  PLACEHOLDER_BLK_RK4

/*
 * Fixed-step RK4 for the Bloch Hamiltonian on Apple Silicon via Metal.
 *
 * Dispatch:  natoms * BLK_RK4 total threads.
 * Buffer layout (matches Python _run_batched call order):
 *   [0]  phi_all   (in/out)  (natoms * NN)  float2  – complex64
 *   [1]  params    (in)      [natoms, nsteps, h_bits]  int32
 *             params[2] holds the float32 step size h reinterpreted as int32
 *             bits (as_type<float> in MSL decodes it).  This avoids passing a
 *             1-element float array, which triggers a NumPy DeprecationWarning
 *             inside metalcompute's scalar-detection heuristic.
 *   [2]  env_full  (in)      (nsteps+1,)  float
 *   [3]  env_half  (in)      (nsteps,)    float
 *   [4]  omegas    (in)      (natoms,)    float
 *   [5]  ep_full_T (in)      (nsteps+1, natoms)  float2  – CPU-computed, f32
 *   [6]  ep_half_T (in)      (nsteps,   natoms)  float2
 *   [7]  eph_full  (in)      (nsteps+1,)  float2
 *   [8]  eph_half  (in)      (nsteps,)    float2
 *   [9]  kinp_full (in)      (nsteps+1, NN)  float2
 *  [10]  kinm_full (in)      (nsteps+1, NN)  float2
 *  [11]  kinp_half (in)      (nsteps,   NN)  float2
 *  [12]  kinm_half (in)      (nsteps,   NN)  float2
 */
kernel void bloch_rk4_metal(
    device       float2*  phi_all   [[ buffer( 0) ]],
    device const int*     params    [[ buffer( 1) ]],
    device const float*   env_full  [[ buffer( 2) ]],
    device const float*   env_half  [[ buffer( 3) ]],
    device const float*   omegas    [[ buffer( 4) ]],
    device const float2*  ep_full_T [[ buffer( 5) ]],
    device const float2*  ep_half_T [[ buffer( 6) ]],
    device const float2*  eph_full  [[ buffer( 7) ]],
    device const float2*  eph_half  [[ buffer( 8) ]],
    device const float2*  kinp_full [[ buffer( 9) ]],
    device const float2*  kinm_full [[ buffer(10) ]],
    device const float2*  kinp_half [[ buffer(11) ]],
    device const float2*  kinm_half [[ buffer(12) ]],
    uint gid [[ thread_position_in_grid         ]],
    uint tid [[ thread_position_in_threadgroup  ]])
{
    /* Four float32 scratch arrays, 1024 slots each (16 KB total).
     * Indexed by tid so that each atom's BLK_RK4 threads occupy
     * a contiguous, non-overlapping slice. */
    threadgroup float arr_a_r[1024];
    threadgroup float arr_a_i[1024];
    threadgroup float arr_b_r[1024];
    threadgroup float arr_b_i[1024];

    int natoms = params[0];
    int nsteps = params[1];
    float h    = as_type<float>(params[2]);  // float32 bits packed in int32
    float h2   = 0.5f * h;
    float h6   = h / 6.0f;

    /* Derive atom and within-atom thread index.
     * n == tid % BLK_RK4 because BLK_RK4 divides the tg_size (1024). */
    int atom = (int)gid / BLK_RK4;
    int n    = (int)tid % BLK_RK4;

    if (atom >= natoms) return;

    /* ── Load initial wavefunction ─────────────────────────────────────── */
    float phi_r = 0.0f, phi_i = 0.0f;
    if (n < NN) {
        float2 v = phi_all[atom * NN + n];
        phi_r = v.x;  phi_i = v.y;
    }
    arr_a_r[tid] = phi_r;
    arr_a_i[tid] = phi_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float omega = omegas[atom];

    /* ── RK4 time-stepping loop ────────────────────────────────────────── */
    for (int j = 0; j < nsteps; j++) {

        /* Detuning phase (transposed table, per-atom, CPU-computed f64→f32) */
        float2 epd0v = ep_full_T[ j      * natoms + atom];
        float2 epdhv = ep_half_T[ j      * natoms + atom];
        float2 epd1v = ep_full_T[(j + 1) * natoms + atom];

        float epd0_re  = epd0v.x,  epd0_im  = epd0v.y;
        float epd_h_re = epdhv.x,  epd_h_im = epdhv.y;
        float epd1_re  = epd1v.x,  epd1_im  = epd1v.y;

        /* Shared phase (same for all atoms) */
        float2 eph0v  = eph_full[j];
        float2 ephh_v = eph_half[j];
        float2 eph1v  = eph_full[j + 1];

        float eph0_re  = eph0v.x,  eph0_im  = eph0v.y;
        float eph_h_re = ephh_v.x, eph_h_im = ephh_v.y;
        float eph1_re  = eph1v.x,  eph1_im  = eph1v.y;

        /* Combined coupling phase: ep_total = ep_delta * eph(t) */
        float ep0_re  = epd0_re  * eph0_re  - epd0_im  * eph0_im;
        float ep0_im  = epd0_re  * eph0_im  + epd0_im  * eph0_re;
        float ep1h_re = epd_h_re * eph_h_re - epd_h_im * eph_h_im;
        float ep1h_im = epd_h_re * eph_h_im + epd_h_im * eph_h_re;
        float ep1_re  = epd1_re  * eph1_re  - epd1_im  * eph1_im;
        float ep1_im  = epd1_re  * eph1_im  + epd1_im  * eph1_re;

        float o0 = env_full[j],  o1 = env_half[j],  o2 = env_full[j + 1];

        float a0 = omega * o0 * 0.5f;
        float c0p_re = -a0 * ep0_im,  c0p_im =  a0 * ep0_re;
        float c0m_re =  a0 * ep0_im,  c0m_im =  a0 * ep0_re;

        float a1 = omega * o1 * 0.5f;
        float c1p_re = -a1 * ep1h_im, c1p_im =  a1 * ep1h_re;
        float c1m_re =  a1 * ep1h_im, c1m_im =  a1 * ep1h_re;

        float a2 = omega * o2 * 0.5f;
        float c2p_re = -a2 * ep1_im,  c2p_im =  a2 * ep1_re;
        float c2m_re =  a2 * ep1_im,  c2m_im =  a2 * ep1_re;

        /* ── k1: reads arr_a, writes arr_b = phi + h/2*k1 ───────────────── */
        float k1r = 0.0f, k1i = 0.0f;
        if (n < NN) {
            float2 kpv = kinp_full[j * NN + n];
            float2 kmv = kinm_full[j * NN + n];
            float kp_re = kpv.x, kp_im = kpv.y;
            float km_re = kmv.x, km_im = kmv.y;
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c0p_re * kp_re - c0p_im * kp_im;
                float t_im = c0p_re * kp_im + c0p_im * kp_re;
                s_re += t_re * arr_a_r[tid + 1] - t_im * arr_a_i[tid + 1];
                s_im += t_re * arr_a_i[tid + 1] + t_im * arr_a_r[tid + 1];
            }
            if (n > 0) {
                float t_re = c0m_re * km_re - c0m_im * km_im;
                float t_im = c0m_re * km_im + c0m_im * km_re;
                s_re += t_re * arr_a_r[tid - 1] - t_im * arr_a_i[tid - 1];
                s_im += t_re * arr_a_i[tid - 1] + t_im * arr_a_r[tid - 1];
            }
            k1r = s_re;  k1i = s_im;
            arr_b_r[tid] = phi_r + h2 * k1r;
            arr_b_i[tid] = phi_i + h2 * k1i;
        } else {
            arr_b_r[tid] = 0.0f;
            arr_b_i[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  /* sync #1 */

        /* ── k2: reads arr_b, writes arr_a = phi + h/2*k2 ───────────────── */
        float k2r = 0.0f, k2i = 0.0f;
        if (n < NN) {
            float2 kpv = kinp_half[j * NN + n];
            float2 kmv = kinm_half[j * NN + n];
            float kp_re = kpv.x, kp_im = kpv.y;
            float km_re = kmv.x, km_im = kmv.y;
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c1p_re * kp_re - c1p_im * kp_im;
                float t_im = c1p_re * kp_im + c1p_im * kp_re;
                s_re += t_re * arr_b_r[tid + 1] - t_im * arr_b_i[tid + 1];
                s_im += t_re * arr_b_i[tid + 1] + t_im * arr_b_r[tid + 1];
            }
            if (n > 0) {
                float t_re = c1m_re * km_re - c1m_im * km_im;
                float t_im = c1m_re * km_im + c1m_im * km_re;
                s_re += t_re * arr_b_r[tid - 1] - t_im * arr_b_i[tid - 1];
                s_im += t_re * arr_b_i[tid - 1] + t_im * arr_b_r[tid - 1];
            }
            k2r = s_re;  k2i = s_im;
            arr_a_r[tid] = phi_r + h2 * k2r;
            arr_a_i[tid] = phi_i + h2 * k2i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  /* sync #2 */

        /* ── k3: reads arr_a, writes arr_b = phi + h*k3 ─────────────────── */
        float k3r = 0.0f, k3i = 0.0f;
        if (n < NN) {
            float2 kpv = kinp_half[j * NN + n];
            float2 kmv = kinm_half[j * NN + n];
            float kp_re = kpv.x, kp_im = kpv.y;
            float km_re = kmv.x, km_im = kmv.y;
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c1p_re * kp_re - c1p_im * kp_im;
                float t_im = c1p_re * kp_im + c1p_im * kp_re;
                s_re += t_re * arr_a_r[tid + 1] - t_im * arr_a_i[tid + 1];
                s_im += t_re * arr_a_i[tid + 1] + t_im * arr_a_r[tid + 1];
            }
            if (n > 0) {
                float t_re = c1m_re * km_re - c1m_im * km_im;
                float t_im = c1m_re * km_im + c1m_im * km_re;
                s_re += t_re * arr_a_r[tid - 1] - t_im * arr_a_i[tid - 1];
                s_im += t_re * arr_a_i[tid - 1] + t_im * arr_a_r[tid - 1];
            }
            k3r = s_re;  k3i = s_im;
            arr_b_r[tid] = phi_r + h * k3r;
            arr_b_i[tid] = phi_i + h * k3i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  /* sync #3 */

        /* ── k4 + phi update: reads arr_b ───────────────────────────────── */
        if (n < NN) {
            float2 kpv = kinp_full[(j + 1) * NN + n];
            float2 kmv = kinm_full[(j + 1) * NN + n];
            float kp_re = kpv.x, kp_im = kpv.y;
            float km_re = kmv.x, km_im = kmv.y;
            float s_re = 0.0f, s_im = 0.0f;
            if (n < NN - 1) {
                float t_re = c2p_re * kp_re - c2p_im * kp_im;
                float t_im = c2p_re * kp_im + c2p_im * kp_re;
                s_re += t_re * arr_b_r[tid + 1] - t_im * arr_b_i[tid + 1];
                s_im += t_re * arr_b_i[tid + 1] + t_im * arr_b_r[tid + 1];
            }
            if (n > 0) {
                float t_re = c2m_re * km_re - c2m_im * km_im;
                float t_im = c2m_re * km_im + c2m_im * km_re;
                s_re += t_re * arr_b_r[tid - 1] - t_im * arr_b_i[tid - 1];
                s_im += t_re * arr_b_i[tid - 1] + t_im * arr_b_r[tid - 1];
            }
            phi_r += h6 * (k1r + 2.0f * k2r + 2.0f * k3r + s_re);
            phi_i += h6 * (k1i + 2.0f * k2i + 2.0f * k3i + s_im);
            arr_a_r[tid] = phi_r;
            arr_a_i[tid] = phi_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  /* sync #4 */
    }

    /* ── Write back ────────────────────────────────────────────────────── */
    if (n < NN) {
        phi_all[atom * NN + n] = float2(phi_r, phi_i);
    }
}
"""

# ── Kernel cache ──────────────────────────────────────────────────────────────

_bloch_metal_cache = {}   # N → (dev, fn)


def _compile_bloch_metal(N):
    import metalcompute as mc
    blk    = _next_pow2(N)
    source = (_METAL_BLOCH_SOURCE_TMPL
              .replace('PLACEHOLDER_NN',      str(N))
              .replace('PLACEHOLDER_BLK_RK4', str(blk)))
    dev = mc.Device()
    fn  = dev.kernel(source).function('bloch_rk4_metal')
    return dev, fn


def _ensure_bloch_metal(N):
    if N not in _bloch_metal_cache:
        _bloch_metal_cache[N] = _compile_bloch_metal(N)
    return _bloch_metal_cache[N]
