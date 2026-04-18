"""Backend implementations for :py:func:`mwave.integrate.propagate`.

- ``'scipy'``  — adaptive ``solve_ivp`` (single-atom only, supports dense output)
- ``'numba'``  — Numba ``@njit(parallel=True)`` with ``prange``  (no external deps)
- ``_fixed_schedule`` — Numba fused DP5 replaying a precomputed step schedule
  across a batch; used by :meth:`NumericBraggInterferometer.compile_numeric_sweep`.
"""

from ._scipy import _run_scipy
from ._numba import (
    _rk45_dp_step,
    _rk45_bloch_adaptive,
)
from ._fixed_schedule import (
    fused_dp5,
    pilot_schedule,
    precompute_step_arrays,
    BUTCHER_C,
)
