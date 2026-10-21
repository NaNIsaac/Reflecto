from collections.abc import Callable, Protocol
from typing import Any
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .schema import Experiment

Residual = Callable[[np.ndarray, np.ndarray], np.ndarray]

class ReflectEngine(Protocol):
    def simulate(self, experiment: Experiment) -> np.ndarray:
        ...

@dataclass
class ParamBinder:
    target_obj: Any
    attr_name: str
    min_val: float = -math.inf
    max_val: float = math.inf

    def __post_init__(self):
        if self.min_val > self.max_val:
            raise AttributeError(f"max_val{self.max_val} should be equal or larger than min_val{self.min_val}")

    @property
    def value(self) -> float:
        return getattr(self.target_obj, self.attr_name)

    @value.setter
    def value(self, val: float):
        setattr(self.target_obj, self.attr_name, val)
    
    def get_bounds(self) -> tuple[float, float]:
        return (self.min_val, self.max_val)


class Fitter:
    def __init__(self, experiment: Experiment, residual: Residual, xray_engine: ReflectEngine):
        self.experiment = experiment
        self.params: list[ParamBinder] = []
        self.residual = residual
        self.engine = xray_engine
    def add_param(self, fit_param: ParamBinder):
        self.params.append(fit_param)

    def _objective(self, param_values: np.ndarray) -> np.ndarray:

        for p, val in zip(self.params, param_values):
            p.value = val

        refl_sim = self.engine.simulate(self.experiment)

        return self.experiment.refl - refl_sim

    def fit(self):
        if not self.params:
            return
        
        x0 = [p.value for p in self.params]
        bounds = (
            [p.min_val for p in self.params],
            [p.max_val for p in self.params]
        )

        result = least_squares(
            self._objective,
            x0,
            bounds=bounds,
            method='trf',
            verbose=2
        )

        return result



@dataclass
class FitStep:
    name: str
    active_params: list[ParamBinder]
    max_iter: int = 100
    tolerance: float = 1e-6

class FitPipeline:
    def __init__(self, fitter):
        pass

if __name__ == "__main__":

    import pandas as pd
    from reflecto.simulate.simulate import param2refl, ParamSet

    from .schema import Instrument, Sample, Layer

    class TempXrayEngine:
        def simulate(self, experiment: Experiment) -> np.ndarray:
            *films, sio2 = experiment.sample.layers
            films_paramsets = [ParamSet(f.thickness, f.roughness, f.sld) for f in films]
            sio2_paramset = ParamSet(ParamSet(sio2.thickness, sio2.roughness, sio2.sld))
            return param2refl(experiment.q, films_paramsets, sio2_paramset)
    
    instrument = Instrument()
    Sample()
    qs = np.linspace(0.01, 3, 300)
    R = pd.read_csv("./test_data/In2O3_Ar_30nm/#1_xrr.dat")
    experiment = Experiment(qs, R, instrument)
    fitter = Fitter()