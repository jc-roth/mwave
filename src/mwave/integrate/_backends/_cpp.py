"""C++ OpenMP backend for bloch_rk4.

Contains:
- C++ source template                         (_CPP_BLOCH_SOURCE_TMPL)
- Build directory path                        (_CPP_BUILD_DIR)
- Compiled library cache                      (_bloch_cpp_libs)
- Compile-on-first-use helper                 (_compile_bloch_cpp)
- Cache-aware ensure helper                   (_ensure_bloch_cpp)
"""

import ctypes as _ctypes
import subprocess as _subprocess
import hashlib as _hashlib
import os as _os


# ── Build directory (relative to the repo root) ───────────────────────────────
# __file__ is mwave/src/mwave/integrate/_backends/_cpp.py
# five ..  levels: _backends → integrate → mwave → src → mwave(pkg root) → IfrEvolveLarge
_EVOLVE_ROOT   = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..', '..'))
_CPP_BUILD_DIR = _os.path.join(_EVOLVE_ROOT, 'src', 'cpp_build')


# ── C++ source template ───────────────────────────────────────────────────────
# Adapted from population/candidates/rk4_cpp_v2.py.
# delta_phases replaced by eph_full/eph_half + t0_d:
#   ep tracks exp(i*delta_i*t_j) only (detuning); phase(t) comes from eph arrays.
#   Combined coupling phase = ep_delta * eph(t_j).
# NN substituted at compile time via string replacement.

_CPP_BLOCH_SOURCE_TMPL = r"""
#include <complex>
#include <cmath>
#include <omp.h>

using cf = std::complex<float>;
using cd = std::complex<double>;

static constexpr int NN = PLACEHOLDER_NN;

extern "C" void rk4_bloch_f32(
    cf*           phi_all,     /* (natoms, NN) complex64, in-place           */
    int           natoms,
    float         h,
    double        t0_d,        /* integration start time                     */
    const float*  env_full,    /* (nsteps+1,) float32 — omega_envelope(t_j)  */
    const float*  env_half,    /* (nsteps,)   float32                        */
    const cf*     eph_full,    /* (nsteps+1,) complex64 — exp(i*phase(t_j))  */
    const cf*     eph_half,    /* (nsteps,)   complex64                      */
    int           nsteps,
    const float*  omegas,      /* (natoms,)   float32 — per-atom Rabi scale  */
    const double* deltas,      /* (natoms,)   float64                        */
    const cf*     kinp_full,   /* (nsteps+1, NN) complex64                   */
    const cf*     kinm_full,
    const cf*     kinp_half,
    const cf*     kinm_half
) {
    const float h2f = 0.5f * h;
    const float h6f = h / 6.0f;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < natoms; i++) {
        const float  omega_i = omegas[i];
        const double delta_i = deltas[i];

        /* ep tracks exp(i*delta_i*t_j) only; phase(t) is in eph_full/eph_half */
        cd ep           = std::exp(cd(0.0, delta_i * t0_d));
        const cd coup_inc   = std::exp(cd(0.0, delta_i * (double)h));
        const cd coup_inc_h = std::exp(cd(0.0, delta_i * (double)h * 0.5));

        cf* __restrict__ phi = phi_all + i * NN;
        cf k1[NN], k2[NN], k3[NN], tmp[NN];

        for (int j = 0; j < nsteps; j++) {
            const float o0 = env_full[j];
            const float o1 = env_half[j];
            const float o2 = env_full[j + 1];

            const cf* kp0 = kinp_full + j       * NN;
            const cf* km0 = kinm_full + j       * NN;
            const cf* kph = kinp_half + j       * NN;
            const cf* kmh = kinm_half + j       * NN;
            const cf* kp1 = kinp_full + (j + 1) * NN;
            const cf* km1 = kinm_full + (j + 1) * NN;

            const cd ep1h_ = ep * coup_inc_h;
            const cd ep1_  = ep * coup_inc;

            /* Detuning phase (f64 precision, truncated to f32) */
            const float epd0_re  = (float)ep.real(),    epd0_im  = (float)ep.imag();
            const float epd1h_re = (float)ep1h_.real(), epd1h_im = (float)ep1h_.imag();
            const float epd1_re  = (float)ep1_.real(),  epd1_im  = (float)ep1_.imag();

            /* Shared phase from pre-evaluated array */
            const float e0_re  = eph_full[j].real(),     e0_im  = eph_full[j].imag();
            const float e1h_re = eph_half[j].real(),     e1h_im = eph_half[j].imag();
            const float e1_re  = eph_full[j + 1].real(), e1_im  = eph_full[j + 1].imag();

            /* Combined: ep_total = ep_delta * eph(t) */
            const float ep0_re  = epd0_re*e0_re   - epd0_im*e0_im;
            const float ep0_im  = epd0_re*e0_im   + epd0_im*e0_re;
            const float ep1h_re = epd1h_re*e1h_re - epd1h_im*e1h_im;
            const float ep1h_im = epd1h_re*e1h_im + epd1h_im*e1h_re;
            const float ep1_re  = epd1_re*e1_re   - epd1_im*e1_im;
            const float ep1_im  = epd1_re*e1_im   + epd1_im*e1_re;

            const float a0 = omega_i * o0 * 0.5f;
            const cf c0p = cf(-a0*ep0_im,   a0*ep0_re);
            const cf c0m = cf( a0*ep0_im,   a0*ep0_re);

            k1[0] = c0p * kp0[0] * phi[1];
            for (int n = 1; n < NN - 1; n++)
                k1[n] = c0p*kp0[n]*phi[n+1] + c0m*km0[n]*phi[n-1];
            k1[NN-1] = c0m * km0[NN-1] * phi[NN-2];
            for (int n = 0; n < NN; n++) tmp[n] = phi[n] + h2f*k1[n];

            const float a1 = omega_i * o1 * 0.5f;
            const cf c1p = cf(-a1*ep1h_im, a1*ep1h_re);
            const cf c1m = cf( a1*ep1h_im, a1*ep1h_re);

            k2[0] = c1p * kph[0] * tmp[1];
            for (int n = 1; n < NN - 1; n++)
                k2[n] = c1p*kph[n]*tmp[n+1] + c1m*kmh[n]*tmp[n-1];
            k2[NN-1] = c1m * kmh[NN-1] * tmp[NN-2];
            for (int n = 0; n < NN; n++) tmp[n] = phi[n] + h2f*k2[n];

            k3[0] = c1p * kph[0] * tmp[1];
            for (int n = 1; n < NN - 1; n++)
                k3[n] = c1p*kph[n]*tmp[n+1] + c1m*kmh[n]*tmp[n-1];
            k3[NN-1] = c1m * kmh[NN-1] * tmp[NN-2];
            for (int n = 0; n < NN; n++) tmp[n] = phi[n] + h*k3[n];

            const float a2 = omega_i * o2 * 0.5f;
            const cf c2p = cf(-a2*ep1_im,  a2*ep1_re);
            const cf c2m = cf( a2*ep1_im,  a2*ep1_re);

            cf k4n;
            k4n = c2p * kp1[0] * tmp[1];
            phi[0] += h6f * (k1[0] + 2.0f*k2[0] + 2.0f*k3[0] + k4n);
            for (int n = 1; n < NN - 1; n++) {
                k4n = c2p*kp1[n]*tmp[n+1] + c2m*km1[n]*tmp[n-1];
                phi[n] += h6f * (k1[n] + 2.0f*k2[n] + 2.0f*k3[n] + k4n);
            }
            k4n = c2m * km1[NN-1] * tmp[NN-2];
            phi[NN-1] += h6f * (k1[NN-1] + 2.0f*k2[NN-1] + 2.0f*k3[NN-1] + k4n);

            ep = ep1_;
        }
    }
}
"""

_bloch_cpp_libs = {}   # N → ctypes lib


import sys as _sys


def _openmp_candidates():
    """Return a list of (compiler, extra_flags) pairs to try, in preference order.

    On macOS, Apple Clang does not ship with OpenMP.  We prefer it with
    Homebrew libomp over falling back to a Homebrew GCC binary, because
    Apple Clang produces better-optimised code for Apple Silicon.
    """
    if _sys.platform != 'darwin':
        return [('g++', ['-fopenmp'])]

    candidates = []

    # 1. Apple Clang + Homebrew libomp (preferred on Apple Silicon)
    for prefix in ('/opt/homebrew', '/usr/local'):
        libomp_inc = _os.path.join(prefix, 'opt', 'libomp', 'include')
        libomp_lib = _os.path.join(prefix, 'opt', 'libomp', 'lib')
        if _os.path.isdir(libomp_inc) and _os.path.isdir(libomp_lib):
            candidates.append(('clang++', [
                f'-I{libomp_inc}', f'-L{libomp_lib}',
                '-Xclang', '-fopenmp', '-lomp',
            ]))
            break  # only add once (first prefix that has libomp)

    # 2. Homebrew GCC (g++-N, newest first)
    for prefix in ('/opt/homebrew', '/usr/local'):
        bin_dir = _os.path.join(prefix, 'bin')
        if not _os.path.isdir(bin_dir):
            continue
        gccs = sorted(
            [f for f in _os.listdir(bin_dir) if f.startswith('g++-')],
            reverse=True,
        )
        for gcc in gccs:
            candidates.append((_os.path.join(bin_dir, gcc), ['-fopenmp']))

    # 3. Plain g++ last (may work if the user has a real GCC on PATH)
    candidates.append(('g++', ['-fopenmp']))
    return candidates


def _try_compile(so_path, cpp_path):
    """Try each (compiler, openmp_flags) candidate until one succeeds.

    Returns ``(cmd, None)`` on success or ``(None, last_stderr)`` on total failure.
    """
    base_flags = ['-O3', '-march=native', '-ffast-math',
                  '-shared', '-fPIC', '-std=c++17']
    last_stderr = ''
    for compiler, omp_flags in _openmp_candidates():
        cmd = [compiler] + base_flags + omp_flags + ['-o', so_path, cpp_path]
        result = _subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return cmd, None
        last_stderr = result.stderr
    return None, last_stderr


def _compile_bloch_cpp(N):
    source   = _CPP_BLOCH_SOURCE_TMPL.replace('PLACEHOLDER_NN', str(N))
    src_hash = _hashlib.md5(source.encode()).hexdigest()[:12]
    _os.makedirs(_CPP_BUILD_DIR, exist_ok=True)
    so_path  = _os.path.join(_CPP_BUILD_DIR, f'rk4_bloch_{N}_{src_hash}.so')

    if not _os.path.exists(so_path):
        cpp_path = so_path.replace('.so', '.cpp')
        with open(cpp_path, 'w') as f:
            f.write(source)

        cmd, stderr = _try_compile(so_path, cpp_path)
        if cmd is None:
            raise RuntimeError(f'C++ compilation failed:\n{stderr}')

    lib = _ctypes.CDLL(so_path)
    lib.rk4_bloch_f32.restype = None
    lib.rk4_bloch_f32.argtypes = [
        _ctypes.c_void_p,   # phi_all (complex64)
        _ctypes.c_int,      # natoms
        _ctypes.c_float,    # h
        _ctypes.c_double,   # t0_d
        _ctypes.c_void_p,   # env_full (float32)
        _ctypes.c_void_p,   # env_half (float32)
        _ctypes.c_void_p,   # eph_full (complex64)
        _ctypes.c_void_p,   # eph_half (complex64)
        _ctypes.c_int,      # nsteps
        _ctypes.c_void_p,   # omegas (float32)
        _ctypes.c_void_p,   # deltas (float64)
        _ctypes.c_void_p,   # kinp_full (complex64)
        _ctypes.c_void_p,   # kinm_full (complex64)
        _ctypes.c_void_p,   # kinp_half (complex64)
        _ctypes.c_void_p,   # kinm_half (complex64)
    ]
    return lib


def _ensure_bloch_cpp(N):
    if N not in _bloch_cpp_libs:
        _bloch_cpp_libs[N] = _compile_bloch_cpp(N)
    return _bloch_cpp_libs[N]
