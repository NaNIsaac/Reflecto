import numpy as np

from .physics import r_e

MATERIAL_DB = {
    "Si": {"rho": 2.33, "Z": 14, "M": 28.0855},
    "SiO2": {"rho": 2.20, "Z": 30, "M": 60.0843},
    "Au": {"rho": 19.32, "Z": 79, "M": 196.967},
    "Cu": {"rho": 8.96, "Z": 29, "M": 63.546},
    "Fe": {"rho": 7.87, "Z": 26, "M": 55.845},
    "Air": {"rho": 0.00, "Z": 0, "M": 1.0},
    "H2O": {"rho": 1.00, "Z": 10, "M": 18.015},
}


def get_sld_from_material(name: str, density: float = None) -> float:
    """
    Return the SLD (10^-6 A^-2) of a material, optionally with a custom density.

    Args:
        name: chemical formula key of MATERIAL_DB (e.g. 'Si', 'Au').
        density: optional density in g/cm^3, used instead of the DB value.
    """
    if name not in MATERIAL_DB:
        raise ValueError(f"Unknown material: {name}. Available: {list(MATERIAL_DB.keys())}")

    mat = MATERIAL_DB[name]
    rho = density if density is not None else mat["rho"]

    N_A = 6.02214076e23

    # Electron density in electrons/cm^3
    # rho_e = (rho * N_A * Z) / M
    rho_e = (rho * N_A * mat["Z"]) / mat["M"]

    # SLD = r_e * rho_e
    # r_e is in Angstrom (about 2.817e-5); rho_e is per cm^3, so multiply by 1e-24
    # to get per Angstrom^3

    sld_val = r_e * rho_e * 1e-24

    # XRR convention: SLD in 10^-6 A^-2
    return sld_val * 1e6


def calculate_sld(density: float, Z: int, M: float) -> float:
    """SLD (10^-6 A^-2) from density, summed atomic number Z and molar mass M."""
    N_A = 6.02214076e23
    rho_e = (density * N_A * Z) / M
    sld = r_e * rho_e * 1e-24
    return sld * 1e6
