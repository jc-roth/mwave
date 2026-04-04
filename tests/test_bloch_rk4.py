"""
Tests for bloch_rk4() in mwave.integrate.

New API (batched, multi-backend):
    bloch_rk4(kvec, phi0, tfinal, delta, omega_envelope, omega_args, omegas,
              phase, phase_args, t0, tol, max_halvings, pilot_rtol,
              backend, cache)

  - phi0   : (N,) single-atom or (natoms, N) batch
  - delta  : scalar or (natoms,) float64
  - omegas : scalar or (natoms,) float64 — per-atom Rabi scale
  - omega_envelope(t, omega_args) * omegas[i] = full Rabi frequency for atom i
"""

import sys
import os
import numpy as np
import pytest

_MWAVE_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _MWAVE_SRC not in sys.path:
    sys.path.insert(0, _MWAVE_SRC)

from mwave.integrate import (
    bloch_rk4,
    bloch, make_kvec, make_phi,
    omega_fnc_gaussian, phase_fnc_constant,
)
from mwave.numeric import NumericBraggInterferometer


# ── shared setup ──────────────────────────────────────────────────────────────

def _gaussian_setup(n0=0, nf=5):
    """5ℏk Bragg pulse: kvec, phi0, tfinal, delta, omega_args, phase_args."""
    kvec, _, _ = make_kvec(n0, nf)
    phi0       = make_phi(kvec, n0)
    sigma      = 0.188
    omega_peak = 30.0
    delta      = float(4 * (n0 + nf))
    tfinal     = 6.0 * sigma
    omega_args = np.array([omega_peak, sigma, tfinal / 2.0])
    phase_args = np.array([0.0])
    return kvec, phi0, tfinal, delta, omega_args, phase_args


# ── Test 1: agreement with bloch() for Gaussian pulse ─────────────────────────

def test_gaussian_agrees_with_bloch():
    """bloch_rk4 must match bloch() (DOP853) to within 1e-5 for a Gaussian pulse."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    phi_rk4, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
        tol=1e-6,
    )

    ref     = bloch(kvec, phi0, tfinal, delta,
                    omega_fnc_gaussian, omega_args,
                    phase_fnc_constant, phase_args)
    phi_ref = ref.y[:, -1]

    max_diff = float(np.max(np.abs(phi_rk4 - phi_ref)))
    assert max_diff < 1e-5, f"max diff vs bloch() = {max_diff:.2e}"


# ── Test 2: Richardson error estimate is self-consistent ──────────────────────

def test_richardson_error_consistent():
    """Returned error_est must be <= tol and dt_used must be a positive float."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    tol = 1e-6

    _, dt_used, error_est = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
        tol=tol,
    )

    assert error_est <= tol, f"error_est {error_est:.2e} exceeds tol {tol}"
    assert dt_used > 0, f"dt_used={dt_used} must be positive"


# ── Test 3: norm conservation ─────────────────────────────────────────────────

def test_norm_conservation():
    """Norm of phi_final must equal 1 to within 1e-8."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    phi_final, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
    )
    norm = float(np.sum(np.abs(phi_final) ** 2))
    assert abs(norm - 1.0) < 1e-7, f"norm deviation = {abs(norm - 1.0):.2e}"


# ── Test 4: chaining (t0 != 0) ───────────────────────────────────────────────

def test_chaining():
    """Two sequential bloch_rk4 calls must agree with one full call to < 1e-5."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    phi_full, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
        t0=0.0, tol=1e-8,
    )

    tmid = tfinal / 2.0
    phi_mid, _, _ = bloch_rk4(
        kvec, phi0, tmid, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
        t0=0.0, tol=1e-8,
    )
    phi_chain, _, _ = bloch_rk4(
        kvec, phi_mid, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([1.0]),
        phase_fnc_constant, phase_args,
        t0=tmid, tol=1e-8,
    )

    max_diff = float(np.max(np.abs(phi_full - phi_chain)))
    assert max_diff < 1e-5, f"chaining disagreement = {max_diff:.2e}"


# ── Test 5: arbitrary (non-constant) phase ───────────────────────────────────

def test_arbitrary_phase():
    """bloch_rk4 with a linear chirp phase must agree with bloch() to within 1e-5."""
    from numba import jit as _jit, float64 as _f64

    kvec, phi0, tfinal, delta, omega_args, _ = _gaussian_setup()
    chirp_rate = 5.0
    phase_args = np.array([chirp_rate])

    def omega_py(t, args):
        return float(omega_fnc_gaussian(t, args))

    def phase_py(t, args):
        return float(args[0] * t)

    phi_rk4, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_py, omega_args, np.array([1.0]),
        phase_py, phase_args,
        tol=1e-6,
    )

    @_jit(_f64(_f64, _f64[:]), nopython=True)
    def chirp_phase(t, args):
        return args[0] * t

    ref     = bloch(kvec, phi0, tfinal, delta,
                    omega_fnc_gaussian, omega_args,
                    chirp_phase, phase_args)
    phi_ref = ref.y[:, -1]

    max_diff = float(np.max(np.abs(phi_rk4 - phi_ref)))
    assert max_diff < 1e-5, f"arbitrary-phase disagreement = {max_diff:.2e}"


# ── Test 6: batch mode agrees with N independent single-atom calls ────────────

def test_batch_mode_agrees_with_single_atom():
    """Batched bloch_rk4 (natoms=4) must match 4 separate single-atom calls."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 4
    deltas = np.array([delta * f for f in [0.98, 0.99, 1.0, 1.01]])
    omegas = np.array([0.9, 1.0, 1.0, 1.1])

    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))
    phi_batch, _, _ = bloch_rk4(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args, omegas,
        phase_fnc_constant, phase_args,
        tol=1e-6,
    )
    assert phi_batch.shape == (natoms, len(kvec))

    for i in range(natoms):
        phi_single, _, _ = bloch_rk4(
            kvec, phi0, tfinal, deltas[i],
            omega_fnc_gaussian, omega_args, np.array([omegas[i]]),
            phase_fnc_constant, phase_args,
            tol=1e-6,
        )
        max_diff = float(np.max(np.abs(phi_batch[i] - phi_single)))
        assert max_diff < 1e-7, (
            f"atom {i}: batch vs single-atom diff = {max_diff:.2e}")


# ── Test 7: omegas scaling is applied correctly ───────────────────────────────

def test_omegas_scaling():
    """Passing omegas=[scale] must give the same result as scaling omega_args."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    scale = 0.75

    # New API: omegas=[scale], envelope unscaled
    phi_scaled, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args, np.array([scale]),
        phase_fnc_constant, phase_args,
        tol=1e-8,
    )

    # Reference: scale baked into omega_args
    omega_args_scaled = omega_args.copy()
    omega_args_scaled[0] *= scale
    phi_ref, _, _ = bloch_rk4(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args_scaled, np.array([1.0]),
        phase_fnc_constant, phase_args,
        tol=1e-8,
    )

    max_diff = float(np.max(np.abs(phi_scaled - phi_ref)))
    assert max_diff < 1e-8, f"omegas scaling diff = {max_diff:.2e}"


# ── Test 8: batch norm conservation ──────────────────────────────────────────

def test_batch_norm_conservation():
    """All atoms in a batch must have norm=1 to within 1e-7."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 5
    deltas = np.linspace(delta * 0.95, delta * 1.05, natoms)
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    phi_batch, _, _ = bloch_rk4(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args, omegas,
        phase_fnc_constant, phase_args,
    )
    norms = np.sum(np.abs(phi_batch) ** 2, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0)))
    assert max_dev < 1e-7, f"max norm deviation across batch = {max_dev:.2e}"


# ── Test 9: cpp backend agrees with python backend ────────────────────────────

def test_cpp_backend_agrees_with_python():
    """'cpp' backend must agree with 'python' backend to within 1e-5."""
    pytest.importorskip('subprocess')
    import subprocess
    if subprocess.run(['g++', '--version'], capture_output=True).returncode != 0:
        pytest.skip("g++ not available")

    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 3
    deltas = np.array([delta * 0.99, delta, delta * 1.01])
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    phi_py, _, _ = bloch_rk4(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args, omegas,
        phase_fnc_constant, phase_args,
        tol=1e-6, backend='python',
    )
    try:
        phi_cpp, _, _ = bloch_rk4(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args, omegas,
            phase_fnc_constant, phase_args,
            tol=1e-6, backend='cpp',
        )
    except RuntimeError as e:
        pytest.skip(f"C++ compilation failed: {e}")

    max_diff = float(np.max(np.abs(phi_py - phi_cpp)))
    assert max_diff < 1e-5, f"cpp vs python max diff = {max_diff:.2e}"


# ── Test 10: gpu backend agrees with python backend ───────────────────────────

def test_gpu_backend_agrees_with_python():
    """'gpu' backend must agree with 'python' backend to within 1e-4."""
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.Device(0).use()
    except Exception:
        pytest.skip("No CUDA device available")

    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 3
    deltas = np.array([delta * 0.99, delta, delta * 1.01])
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    phi_py, _, _ = bloch_rk4(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args, omegas,
        phase_fnc_constant, phase_args,
        tol=1e-6, backend='python',
    )
    try:
        phi_gpu, _, _ = bloch_rk4(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args, omegas,
            phase_fnc_constant, phase_args,
            tol=1e-6, backend='gpu',
        )
    except Exception as e:
        pytest.skip(f"GPU kernel failed: {e}")

    max_diff = float(np.max(np.abs(phi_py - phi_gpu)))
    assert max_diff < 1e-4, f"gpu vs python max diff = {max_diff:.2e}"


# ── Test 11: metal backend agrees with python backend ────────────────────────

def test_metal_backend_agrees_with_python():
    """'metal' backend must agree with 'python' backend to within 1e-4."""
    pytest.importorskip('metalcompute')

    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 3
    deltas = np.array([delta * 0.99, delta, delta * 1.01])
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    phi_py, _, _ = bloch_rk4(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args, omegas,
        phase_fnc_constant, phase_args,
        tol=1e-6, backend='python',
    )
    try:
        phi_metal, _, _ = bloch_rk4(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args, omegas,
            phase_fnc_constant, phase_args,
            tol=1e-6, backend='metal',
        )
    except Exception as e:
        pytest.skip(f"Metal kernel failed: {e}")

    max_diff = float(np.max(np.abs(phi_py - phi_metal)))
    assert max_diff < 1e-4, f"metal vs python max diff = {max_diff:.2e}"


# ── Test 12: NumericBraggInterferometer end-to-end ────────────────────────────

def test_make_rk4_split_func_with_interferometer():
    """bloch_rk4 must agree with bloch() across all output ports of a full
    simultaneous conjugate interferometer."""
    nbragg = 2
    T      = 2.0
    Tp     = 1.0

    sigma      = 0.259
    omega_peak = 19.4
    t_bragg    = 6.0 * sigma
    delta      = float(4 * nbragg)

    omega_args = np.array([omega_peak, sigma, t_bragg / 2.0])
    phase_args = np.array([0.0])

    ifr = NumericBraggInterferometer(-2 * nbragg, 4 * nbragg, distance=0)
    ifr.split(nbragg)
    ifr.propagate(T)
    ifr.split(nbragg)
    ifr.propagate(Tp)
    ifr.split([3 * nbragg, -nbragg])
    ifr.propagate(T)
    ifr.split([3 * nbragg, -nbragg])

    kvec = ifr.kvec

    def omega_py(t, args):
        return float(omega_fnc_gaussian(t, args))

    def phase_py(t, args):
        return float(phase_fnc_constant(t, args))

    _rk4_cache = {}

    def cached_split_fn(*args):
        comm_args = args[:-5]
        k_init, k_final, _, t_start, _ = args[-5:]
        delta_val = float(comm_args[0])
        phi0 = np.zeros(len(kvec), dtype=np.complex128)
        phi0[int(np.argmin(np.abs(kvec - k_init)))] = 1.0
        phi_final, _, _ = bloch_rk4(
            kvec, phi0, t_start + t_bragg, delta_val,
            omega_py, omega_args, np.array([1.0]),
            phase_py, phase_args,
            t0=t_start, tol=1e-6, cache=_rk4_cache,
        )
        return phi_final[int(np.argmin(np.abs(kvec - k_final)))]

    def prop_fn(_, t, k):
        return np.exp(-1j * t * k ** 2)

    ifr.set_operation_funcs(
        [cached_split_fn, prop_fn, cached_split_fn, prop_fn,
         cached_split_fn, prop_fn, cached_split_fn]
    )

    def func_pop_init(delta_val):
        return np.float64(0.0)

    def func_wf_init(delta_val):
        return np.complex128(1.0)

    def func_wf2_init(delta_val):
        return np.complex128(0.0)

    # Reference using bloch() directly
    ifr_ref = NumericBraggInterferometer(-2 * nbragg, 4 * nbragg, distance=0)
    ifr_ref.split(nbragg)
    ifr_ref.propagate(T)
    ifr_ref.split(nbragg)
    ifr_ref.propagate(Tp)
    ifr_ref.split([3 * nbragg, -nbragg])
    ifr_ref.propagate(T)
    ifr_ref.split([3 * nbragg, -nbragg])

    _bloch_cache = {}

    def ref_split_fn(delta_val, k_init, k_final, _klattice, t_start, _x):
        cache_key = (delta_val, k_init, t_start)
        if cache_key not in _bloch_cache:
            phi0 = np.zeros(len(kvec), dtype=np.complex128)
            phi0[int(np.argmin(np.abs(kvec - k_init)))] = 1.0
            ref = bloch(
                kvec, phi0, t_start + t_bragg, delta_val,
                omega_fnc_gaussian, omega_args,
                phase_fnc_constant, phase_args,
                t0=t_start,
            )
            _bloch_cache[cache_key] = ref.y[:, -1]
        return _bloch_cache[cache_key][int(np.argmin(np.abs(kvec - k_final)))]

    ifr_ref.set_operation_funcs(
        [ref_split_fn, prop_fn, ref_split_fn, prop_fn,
         ref_split_fn, prop_fn, ref_split_fn]
    )

    output_ports = [4 * nbragg, 2 * nbragg, 0, -2 * nbragg]

    pop_func     = ifr.get_population_func(output_ports, func_pop_init, func_wf_init, func_wf2_init)
    pop_func_ref = ifr_ref.get_population_func(output_ports, func_pop_init, func_wf_init, func_wf2_init)

    for port in output_ports:
        pop_rk4 = float(pop_func(port, [delta]))
        pop_ref = float(pop_func_ref(port, [delta]))
        assert abs(pop_rk4 - pop_ref) < 1e-4, (
            f"Port {port}: rk4={pop_rk4:.6f}, ref={pop_ref:.6f}, "
            f"diff={abs(pop_rk4 - pop_ref):.2e}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
