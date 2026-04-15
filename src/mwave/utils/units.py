import numpy as np

# Cs-133 D2 line single-photon recoil constants
OMEGA_R = 2.0663e3 * 2 * np.pi  # recoil angular frequency [rad/s]
V_R = 3.5225e-3                  # recoil velocity [m/s]
L_R = V_R / OMEGA_R              # recoil length [m]
C = 299_792_458.0                # speed of light [m/s]

class RecoilUnits:
    """Converts between SI units and the single-photon recoil unit system for Cs-133 on the D2 line. Time is in units of :math:`1/\\omega_r`, length in :math:`v_r/\\omega_r`, and velocity in :math:`v_r`.

    :param omega_r: Recoil angular frequency in rad/s. Defaults to the Cs-133 D2 value.
    :param v_r: Recoil velocity in m/s. Defaults to the Cs-133 D2 value.

    Example

    >>> from mwave.utils.units import recoil
    >>> recoil.time(4.5e-3)        # 4.5 ms interrogation time to recoil units
    58.4232561010133
    >>> recoil.length(6.2e-3)      # 6.2 mm beam waist to recoil units
    22851.45889606703
    >>> recoil.velocity(299792458) # speed of light to recoil units
    85107866004.25833
    >>> recoil.time(58.4232561010133, inverse=True)  # back to SI seconds
    0.0045

    """

    def __init__(self, omega_r=OMEGA_R, v_r=V_R):
        self.omega_r = omega_r
        self.v_r = v_r
        self.L_r = v_r / omega_r

    def time(self, value, inverse=False):
        """Convert time between SI seconds and recoil units.

        :param value: The value to convert.
        :param inverse: If ``True``, convert from recoil units to SI. Defaults to ``False``.
        :returns: The converted value."""
        return value / self.omega_r if inverse else value * self.omega_r

    def length(self, value, inverse=False):
        """Convert length between SI metres and recoil units.

        :param value: The value to convert.
        :param inverse: If ``True``, convert from recoil units to SI. Defaults to ``False``.
        :returns: The converted value."""
        return value * self.L_r if inverse else value / self.L_r

    def velocity(self, value, inverse=False):
        """Convert velocity between SI m/s and recoil units.

        :param value: The value to convert.
        :param inverse: If ``True``, convert from recoil units to SI. Defaults to ``False``.
        :returns: The converted value."""
        return value * self.v_r if inverse else value / self.v_r

    def frequency(self, value, inverse=False):
        """Convert frequency between SI Hz and recoil units.

        :param value: The value to convert.
        :param inverse: If ``True``, convert from recoil units to SI. Defaults to ``False``.
        :returns: The converted value."""
        f_r = self.omega_r / (2 * np.pi)
        return value * f_r if inverse else value / f_r

    def angular_frequency(self, value, inverse=False):
        """Convert angular frequency between SI rad/s and recoil units.

        :param value: The value to convert.
        :param inverse: If ``True``, convert from recoil units to SI. Defaults to ``False``.
        :returns: The converted value."""
        return value * self.omega_r if inverse else value / self.omega_r


recoil = RecoilUnits()
