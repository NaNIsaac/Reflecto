import numpy as np
from scipy.special import erf

from ..simulate.simulate import ParamSet


def calc_profile(layers: list, z_step: float = 0.5, padding: float = 30.0):
    """
    Compute the SLD profile versus depth z.

    Args:
        layers: ParamSets ordered from the top layer to the bottom layer
    """
    total_thickness = sum(L.thickness for L in layers)
    z_max = total_thickness + padding
    z = np.arange(-padding, z_max, z_step)

    current_sld = layers[0].sld
    sld_profile = np.full_like(z, current_sld, dtype=float)

    current_interface_z = 0.0

    for i in range(len(layers) - 1):
        top_layer = layers[i]
        bot_layer = layers[i + 1]

        # Each interface takes the roughness of the layer below it
        sigma = bot_layer.roughness
        if sigma < 1e-3:
            sigma = 1e-3

        sld_diff = bot_layer.sld - top_layer.sld

        # Error Function Smoothing
        transition = 0.5 * (1 + erf((z - current_interface_z) / (np.sqrt(2) * sigma)))
        sld_profile += sld_diff * transition

        # Next interface lies one top-layer thickness lower; the ambient layer needs
        # thickness 0 so the first interface sits at z=0
        d = top_layer.thickness
        current_interface_z += d

    return z, sld_profile
