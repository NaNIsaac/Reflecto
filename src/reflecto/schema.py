from dataclasses import dataclass, field

import numpy as np

RE_ANGSTROM = 2.81794032e-5

@dataclass
class Instrument:
    wavelength: float = 1.54
    I0: float = 1.0
    Ibkg: float = 1e-10
    res: float = 0.0001

@dataclass
class Layer:
    thickness: float  # [Å]
    roughness: float  # [Å]
    sld: complex  # [10^-6 Å^-2]
    f: complex = field(default=complex(1, 0), repr=False)

    @property
    def density(self) -> float:
        """
        Unit: [atoms / Å^3]
        Formula:
            SLD = r_e * rho * f
            rho = SLD / (r_e * f)
        """
        if self.f.real == 0:
            return 0.0

        real_sld_value = self.sld.real * 1e-6 
        
        return real_sld_value / (RE_ANGSTROM * self.f.real)

AIR = Layer(thickness=0, roughness=0, sld=complex(0, 0))
SI_SUBSTRATE = Layer(thickness=0, roughness=2.0, sld=complex(20.1, 0.46))

@dataclass
class Sample:
    layers: list[Layer]
    Ambient: Layer
    Substrate: Layer

@dataclass
class Experiment:
    q: np.ndarray
    refl: np.ndarray
    instrument: Instrument
    sample: Sample
