from typing import Final, overload

import numpy as np
import scipy.constants as const
from numpy.typing import NDArray

r_e: Final[float] = 2.8179403227e-5


@overload
def tth2q(tth: float, wavelen: float = 1.54) -> float: ...


@overload
def tth2q(tth: NDArray[np.float64], wavelen: float = 1.54) -> NDArray[np.float64]: ...


def tth2q(tth, wavelen=1.54):
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    Formula: q = (4 * pi / lambda) * sin(theta)
    """
    th_rad = np.deg2rad(0.5 * tth)
    result = (4 * np.pi / wavelen) * np.sin(th_rad)

    if isinstance(tth, (int, float)):
        return float(result)
    return result


def sld_to_density(sld: float | np.ndarray) -> float | np.ndarray:
    """
    Convert SLD (10^-6 A^-2) to electron density (electrons/cm^3).
    Formula: rho_e = (SLD * 1e-6) / r_e * 1e24
    """
    return (sld * 1e-6 / r_e) * 1e24


def calc_theoretical_sld(density: float, total_z: int, molar_mass: float) -> tuple[float, float]:
    """
    Compute the theoretical SLD and electron density from density and chemical composition.

    Args:
        density (float): density (g/cm^3)
        total_z (int): total atomic number per molecule (electrons per molecule)
        molar_mass (float): molar mass (g/mol)

    Returns:
        Tuple[float, float]: (SLD [x10^-6 A^-2], Electron Density [e/A^3])
    """
    # Electron density in electrons/A^3
    # rho_e = (density * N_A * Total_Z) / Molar_Mass * (1e-24 cm^3/A^3)
    rho_e = (density * const.Avogadro * total_z) / molar_mass * 1e-24

    # Scaled by 1e6 (10^-6 A^-2) for compatibility with XRR tools
    # SLD = r_e * rho_e
    sld_val = r_e * rho_e * 1e6

    return sld_val, rho_e


def sld_to_material_properties(
    sld: float, total_z: float, molar_mass: float
) -> tuple[float, float]:
    """
    Convert fitted SLD to Physical Densities using Chemical Information (Form Factor).

    Physics:
        SLD = r_e * (rho_mass * N_A / M) * f_1
        (Assuming f_1 approx Z for standard XRR)

    Args:
        sld (float): SLD value in x10^-6 A^-2
        total_z (float): Total Form Factor (Total Z of the molecule)
        molar_mass (float): Molar Mass (g/mol)

    Returns:
        Tuple[float, float]: (Electron Density [e/A^3], Mass Density [g/cm^3])
    """
    # SLD (10^-6 A^-2) -> Convert to A^-2: sld * 1e-6
    # rho_n [molecules/A^3] = SLD_raw / (r_e * total_z)
    sld_raw = sld * 1e-6
    rho_n = sld_raw / (r_e * total_z)

    rho_e = rho_n * total_z

    # rho_mass [g/cm^3] = rho_n [A^-3] * M [g/mol] / N_A * 1e24 [A^3/cm^3]
    rho_mass = (rho_n * molar_mass / const.Avogadro) * 1e24

    return rho_e, rho_mass
