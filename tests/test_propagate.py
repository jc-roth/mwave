"""
Tests for propagate() in mwave.integrate.

Unified API:
    propagate(kvec, phi0, tfinal, delta, omega, omega_args, phase, phase_args,
              omegas=None, t0=0.0, backend=None, dense=False, method='DOP853',
              atol=1e-10, rtol=1e-10, max_step=0.1, transformed=False,
              Gamma_sps=None, tol=1e-6, max_halvings=6, pilot_rtol=1e-8,
              cache=None)

  - phi0   : (N,) single-atom or (natoms, N) batch
  - delta  : scalar or (natoms,) float64
  - omegas : scalar or (natoms,) float64 — per-atom Rabi scale (default 1.0)
  - backend: 'scipy', 'numba', 'cpp', 'gpu', 'metal', or None (auto)
"""

import numpy as np
import pytest

from mwave.integrate import (
    propagate, PropagateResult, make_kvec, make_phi,
    omega_fnc_gaussian, phase_fnc_constant,
)
from mwave.numeric import NumericBraggInterferometer


# ── shared setup ──────────────────────────────────────────────────────────────

def _gaussian_setup(n0=0, nf=5):
    """5hk Bragg pulse: kvec, phi0, tfinal, delta, omega_args, phase_args."""
    kvec, _, _ = make_kvec(n0, nf)
    phi0       = make_phi(kvec, n0)
    sigma      = 0.188
    omega_peak = 30.0
    delta      = float(4 * (n0 + nf))
    tfinal     = 6.0 * sigma
    omega_args = np.array([omega_peak, sigma, tfinal / 2.0])
    phase_args = np.array([0.0])
    return kvec, phi0, tfinal, delta, omega_args, phase_args


# ── Test 1: agreement between python and scipy backends ──────────────────────

def test_gaussian_agrees_with_scipy():
    """propagate(backend='numba') must match propagate(backend='scipy') to
    within 1e-5 for a Gaussian pulse."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    res_rk = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, tol=1e-6, backend='numba',
    )

    res_scipy = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        backend='scipy',
    )

    assert isinstance(res_rk, PropagateResult)
    assert isinstance(res_scipy, PropagateResult)
    max_diff = float(np.max(np.abs(res_rk.phi_final - res_scipy.phi_final)))
    assert max_diff < 1e-5, f"max diff vs scipy = {max_diff:.2e}"


# ── Test 2: error estimate is self-consistent ────────────────────────────────

def test_error_consistent():
    """Returned error must be <= tol and dt must be a positive float."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    tol = 1e-6

    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, tol=tol, backend='numba',
    )

    assert res.error <= tol, f"error {res.error:.2e} exceeds tol {tol}"
    assert res.dt > 0, f"dt={res.dt} must be positive"
    assert res.scipy_sol is None


# ── Test 3: norm conservation ────────────────────────────────────────────────

def test_norm_conservation():
    """Norm of phi_final must equal 1 to within 1e-7."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, backend='numba',
    )
    norm = float(np.sum(np.abs(res.phi_final) ** 2))
    assert abs(norm - 1.0) < 1e-7, f"norm deviation = {abs(norm - 1.0):.2e}"


# ── Test 4: chaining (t0 != 0) ──────────────────────────────────────────────

def test_chaining():
    """Two sequential propagate calls must agree with one full call to < 1e-5."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()

    res_full = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, t0=0.0, tol=1e-8, backend='numba',
    )

    tmid = tfinal / 2.0
    res_mid = propagate(
        kvec, phi0, tmid, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, t0=0.0, tol=1e-8, backend='numba',
    )
    res_chain = propagate(
        kvec, res_mid.phi_final, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, t0=tmid, tol=1e-8, backend='numba',
    )

    max_diff = float(np.max(np.abs(res_full.phi_final - res_chain.phi_final)))
    assert max_diff < 1e-5, f"chaining disagreement = {max_diff:.2e}"


# ── Test 5: arbitrary (non-constant) phase ───────────────────────────────────

def test_arbitrary_phase():
    """propagate with a linear chirp phase must agree between python and scipy
    backends to within 1e-5."""
    from numba import jit as _jit, float64 as _f64

    kvec, phi0, tfinal, delta, omega_args, _ = _gaussian_setup()
    chirp_rate = 5.0
    phase_args = np.array([chirp_rate])

    def omega_py(t, args):
        return float(omega_fnc_gaussian(t, args))

    def phase_py(t, args):
        return float(args[0] * t)

    res_rk = propagate(
        kvec, phi0, tfinal, delta,
        omega_py, omega_args,
        phase_py, phase_args,
        omegas=1.0, tol=1e-6, backend='numba',
    )

    @_jit(_f64(_f64, _f64[:]), nopython=True)
    def chirp_phase(t, args):
        return args[0] * t

    res_scipy = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        chirp_phase, phase_args,
        backend='scipy',
    )

    max_diff = float(np.max(np.abs(res_rk.phi_final - res_scipy.phi_final)))
    assert max_diff < 1e-5, f"arbitrary-phase disagreement = {max_diff:.2e}"


# ── Test 6: batch mode agrees with N independent single-atom calls ───────────

def test_batch_mode_agrees_with_single_atom():
    """Batched propagate (natoms=4) must match 4 separate single-atom calls."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 4
    deltas = np.array([delta * f for f in [0.98, 0.99, 1.0, 1.01]])
    omegas = np.array([0.9, 1.0, 1.0, 1.1])

    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))
    res_batch = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, tol=1e-6, backend='numba',
    )
    assert res_batch.phi_final.shape == (natoms, len(kvec))

    for i in range(natoms):
        res_single = propagate(
            kvec, phi0, tfinal, deltas[i],
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            omegas=omegas[i], tol=1e-6, backend='numba',
        )
        max_diff = float(np.max(np.abs(res_batch.phi_final[i] - res_single.phi_final)))
        assert max_diff < 1e-7, (
            f"atom {i}: batch vs single-atom diff = {max_diff:.2e}")


# ── Test 7: omegas scaling is applied correctly ──────────────────────────────

def test_omegas_scaling():
    """Passing omegas=scale must give the same result as scaling omega_args."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    scale = 0.75

    # Omegas scaling
    res_scaled = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=scale, tol=1e-8, backend='numba',
    )

    # Reference: scale baked into omega_args
    omega_args_scaled = omega_args.copy()
    omega_args_scaled[0] *= scale
    res_ref = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args_scaled,
        phase_fnc_constant, phase_args,
        omegas=1.0, tol=1e-8, backend='numba',
    )

    max_diff = float(np.max(np.abs(res_scaled.phi_final - res_ref.phi_final)))
    assert max_diff < 1e-8, f"omegas scaling diff = {max_diff:.2e}"


# ── Test 8: batch norm conservation ──────────────────────────────────────────

def test_batch_norm_conservation():
    """All atoms in a batch must have norm=1 to within 1e-7."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 5
    deltas = np.linspace(delta * 0.95, delta * 1.05, natoms)
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    res = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, backend='numba',
    )
    norms = np.sum(np.abs(res.phi_final) ** 2, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0)))
    assert max_dev < 1e-7, f"max norm deviation across batch = {max_dev:.2e}"


# ── Test 9: cpp backend agrees with python backend ──────────────────────────

def test_cpp_backend_agrees_with_python():
    """'cpp' backend must agree with 'numba' backend to within 1e-5."""
    pytest.importorskip('subprocess')
    import subprocess
    if subprocess.run(['g++', '--version'], capture_output=True).returncode != 0:
        pytest.skip("g++ not available")

    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 3
    deltas = np.array([delta * 0.99, delta, delta * 1.01])
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    res_py = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, tol=1e-6, backend='numba',
    )
    try:
        res_cpp = propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            omegas=omegas, tol=1e-6, backend='cpp',
        )
    except RuntimeError as e:
        pytest.skip(f"C++ compilation failed: {e}")

    max_diff = float(np.max(np.abs(res_py.phi_final - res_cpp.phi_final)))
    assert max_diff < 1e-5, f"cpp vs python max diff = {max_diff:.2e}"


# ── Test 10: gpu backend agrees with python backend ─────────────────────────

def test_gpu_backend_agrees_with_python():
    """'gpu' backend must agree with 'numba' backend to within 1e-4."""
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

    res_py = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, tol=1e-6, backend='numba',
    )
    try:
        res_gpu = propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            omegas=omegas, tol=1e-6, backend='gpu',
        )
    except Exception as e:
        pytest.skip(f"GPU kernel failed: {e}")

    max_diff = float(np.max(np.abs(res_py.phi_final - res_gpu.phi_final)))
    assert max_diff < 1e-4, f"gpu vs python max diff = {max_diff:.2e}"


# ── Test 11: metal backend agrees with python backend ───────────────────────

def test_metal_backend_agrees_with_python():
    """'metal' backend must agree with 'numba' backend to within 1e-4."""
    pytest.importorskip('metalcompute')

    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    natoms = 3
    deltas = np.array([delta * 0.99, delta, delta * 1.01])
    omegas = np.ones(natoms)
    phi0_batch = np.tile(phi0[np.newaxis, :], (natoms, 1))

    res_py = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, tol=1e-6, backend='numba',
    )
    try:
        res_metal = propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            omegas=omegas, tol=1e-6, backend='metal',
        )
    except Exception as e:
        pytest.skip(f"Metal kernel failed: {e}")

    max_diff = float(np.max(np.abs(res_py.phi_final - res_metal.phi_final)))
    assert max_diff < 1e-4, f"metal vs python max diff = {max_diff:.2e}"


# ── Test 12: NumericBraggInterferometer end-to-end ──────────────────────────

def test_propagate_with_interferometer():
    """propagate must agree with scipy backend across all output ports of a full
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

    _rk_cache = {}

    def cached_split_fn(*args):
        comm_args = args[:-5]
        k_init, k_final, _, t_start, _ = args[-5:]
        delta_val = float(comm_args[0])
        phi0 = np.zeros(len(kvec), dtype=np.complex128)
        phi0[int(np.argmin(np.abs(kvec - k_init)))] = 1.0
        res = propagate(
            kvec, phi0, t_start + t_bragg, delta_val,
            omega_py, omega_args,
            phase_py, phase_args,
            omegas=1.0, t0=t_start, tol=1e-6, backend='numba',
            cache=_rk_cache,
        )
        return res.phi_final[int(np.argmin(np.abs(kvec - k_final)))]

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

    # Reference using scipy backend
    ifr_ref = NumericBraggInterferometer(-2 * nbragg, 4 * nbragg, distance=0)
    ifr_ref.split(nbragg)
    ifr_ref.propagate(T)
    ifr_ref.split(nbragg)
    ifr_ref.propagate(Tp)
    ifr_ref.split([3 * nbragg, -nbragg])
    ifr_ref.propagate(T)
    ifr_ref.split([3 * nbragg, -nbragg])

    _scipy_cache = {}

    def ref_split_fn(delta_val, k_init, k_final, _klattice, t_start, _x):
        cache_key = (delta_val, k_init, t_start)
        if cache_key not in _scipy_cache:
            phi0 = np.zeros(len(kvec), dtype=np.complex128)
            phi0[int(np.argmin(np.abs(kvec - k_init)))] = 1.0
            res = propagate(
                kvec, phi0, t_start + t_bragg, delta_val,
                omega_fnc_gaussian, omega_args,
                phase_fnc_constant, phase_args,
                t0=t_start, backend='scipy',
            )
            _scipy_cache[cache_key] = res.phi_final
        return _scipy_cache[cache_key][int(np.argmin(np.abs(kvec - k_final)))]

    ifr_ref.set_operation_funcs(
        [ref_split_fn, prop_fn, ref_split_fn, prop_fn,
         ref_split_fn, prop_fn, ref_split_fn]
    )

    output_ports = [4 * nbragg, 2 * nbragg, 0, -2 * nbragg]

    pop_func     = ifr.get_population_func(output_ports, func_pop_init, func_wf_init, func_wf2_init)
    pop_func_ref = ifr_ref.get_population_func(output_ports, func_pop_init, func_wf_init, func_wf2_init)

    for port in output_ports:
        pop_rk  = float(pop_func(port, [delta]))
        pop_ref = float(pop_func_ref(port, [delta]))
        assert abs(pop_rk - pop_ref) < 1e-4, (
            f"Port {port}: rk={pop_rk:.6f}, ref={pop_ref:.6f}, "
            f"diff={abs(pop_rk - pop_ref):.2e}"
        )


# ── Test 13: validation errors ──────────────────────────────────────────────

def test_batch_scipy_raises():
    """backend='scipy' with batch input must raise ValueError."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    phi0_batch = np.tile(phi0[np.newaxis, :], (2, 1))
    deltas = np.array([delta, delta])
    with pytest.raises(ValueError, match="backend='scipy'"):
        propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            backend='scipy',
        )


def test_batch_dense_raises():
    """dense=True with batch input must raise ValueError."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    phi0_batch = np.tile(phi0[np.newaxis, :], (2, 1))
    deltas = np.array([delta, delta])
    with pytest.raises(ValueError, match="dense"):
        propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            dense=True,
        )


def test_dense_non_scipy_raises():
    """dense=True with backend='numba' must raise ValueError."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    with pytest.raises(ValueError, match="dense"):
        propagate(
            kvec, phi0, tfinal, delta,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            dense=True, backend='numba',
        )


def test_shape_mismatch_raises():
    """Mismatched omegas/delta shapes must raise ValueError."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    phi0_batch = np.tile(phi0[np.newaxis, :], (3, 1))
    deltas = np.array([delta, delta, delta])
    with pytest.raises(ValueError, match="omegas has length"):
        propagate(
            kvec, phi0_batch, tfinal, deltas,
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            omegas=np.ones(5),  # wrong length
        )


# ── Test 14: PropagateResult attributes ─────────────────────────────────────

def test_result_scipy_attributes():
    """scipy result must have scipy_sol set and dt/error as None."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        backend='scipy',
    )
    assert res.scipy_sol is not None
    assert res.dt is None
    assert res.error is None
    assert res.phi_final.shape == (len(kvec),)
    assert np.array_equal(res.kvec, kvec)


def test_result_rk45_attributes():
    """RK45 result must have dt/error set and scipy_sol as None."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, backend='numba',
    )
    assert res.scipy_sol is None
    assert res.dt is not None
    assert res.error is not None
    assert res.phi_final.shape == (len(kvec),)


# ── Test 15: auto backend selection ─────────────────────────────────────────

def test_auto_backend_single_atom():
    """Default backend for single-atom input should use scipy."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
    )
    assert res.scipy_sol is not None, "single-atom default should use scipy"


def test_auto_backend_batch():
    """Default backend for batch input should use python (RK45)."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    phi0_batch = np.tile(phi0[np.newaxis, :], (2, 1))
    deltas = np.array([delta, delta])
    res = propagate(
        kvec, phi0_batch, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
    )
    assert res.scipy_sol is None, "batch default should use python backend"
    assert res.dt is not None


# ── Test 16: population() convenience method ─────────────────────────────────

def test_population_method():
    """population(k) must return correct |amplitude|^2."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        backend='scipy',
    )
    # population at k=0 (initial state)
    idx0 = int(np.argmin(np.abs(kvec - 0)))
    expected = float(np.abs(res.phi_final[idx0]) ** 2)
    assert abs(res.population(0) - expected) < 1e-15


# ── Test 17: plot() raises without scipy_sol ─────────────────────────────────

def test_plot_raises_without_scipy():
    """plot() must raise ValueError when scipy_sol is None."""
    kvec, phi0, tfinal, delta, omega_args, phase_args = _gaussian_setup()
    res = propagate(
        kvec, phi0, tfinal, delta,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=1.0, backend='numba',
    )
    with pytest.raises(ValueError, match="plot.*requires.*scipy"):
        res.plot()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
