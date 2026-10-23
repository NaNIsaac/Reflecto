"""The GenX script that GenXScriptBuilder writes: every fitted instrument parameter must reach the simulation.

Reflecto#2: set_ibkg was a fitted parameter, but the Sim() hook did not copy it into the instrument,
so the background stayed at its starting value for the whole fit.
"""

import numpy as np
import pytest
from genx.data import DataList, DataSet
from genx.model import Model

from reflecto.engines.script_builder import GenXScriptBuilder

INIT = {
    "f_d": 300.0,
    "f_sig": 4.0,
    "f_sld": 40.0,
    "s_d": 15.0,
    "s_sig": 3.0,
    "s_sld": 18.9,
    "i0": 1.0,
    "ibkg": 1e-7,
    "s_len": 10.0,
    "beam_w": 0.1,
}


@pytest.fixture
def model():
    ds = DataSet(name="test")  # as in reflecto.engines.fitting._initialize_genx_model
    ds.x_raw = np.linspace(0.01, 0.5, 300)
    ds.y_raw = np.ones_like(ds.x_raw)
    ds.error_raw = np.full_like(ds.x_raw, 0.1)
    ds.run_command()
    m = Model()
    m.data = DataList([ds])
    m.set_script(GenXScriptBuilder().build(INIT))
    m.compile_script()
    return m


def simulate(model, **values):
    v = model.script_module.v
    for name, value in values.items():
        getattr(v, f"set_{name}")(value)
    # compile=True would rerun the script and reset v; a fit does not recompile either
    model.simulate(compile=False)
    return np.asarray(model.data[0].y_sim, float)


def test_background_reaches_the_simulation(model):
    low = simulate(model, ibkg=1e-7)
    high = simulate(model, ibkg=1e-5)
    # in the high-q tail the curve is the background: it must follow the parameter
    assert high[-1] - low[-1] == pytest.approx(1e-5 - 1e-7, rel=1e-6)


def test_intensity_scale_reaches_the_simulation(model):
    one = simulate(model, i0=1.0)
    two = simulate(model, i0=2.0)
    assert two[0] / one[0] == pytest.approx(2.0, rel=1e-6)
