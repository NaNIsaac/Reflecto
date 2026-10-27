"""Reflecto#3: the GenX model must be Si / SiO2 / film, the stack the bundled weights were trained on."""

from types import SimpleNamespace

import numpy as np
import pytest
from genx.data import DataList, DataSet
from genx.model import Model

from reflecto.engines.fitting import GenXFitter, XRRConfig
from reflecto.engines.script_builder import GenXScriptBuilder

SIO2, SI = 18.8, 20.07  # the script's fixed starting oxide SLD and substrate SLD


def parratt(q, layers):
    """|r|^2 for [(sld 1e-6/A^2, thickness, roughness of its top interface)], ambient first."""
    sld = np.array([lay[0] for lay in layers], complex) * 1e-6
    kz = np.sqrt((q[:, None] / 2) ** 2 - 4 * np.pi * (sld[None, :] - sld[0]) + 0j)
    x = np.zeros(len(q), complex)
    for j in range(len(layers) - 2, -1, -1):
        k1, k2 = kz[:, j], kz[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * layers[j + 1][2] ** 2)
        d = layers[j + 1][1] if j + 1 < len(layers) - 1 else 0.0
        p = np.exp(2j * k2 * d)
        x = (r + x * p) / (1 + r * x * p)
    return np.abs(x) ** 2


def test_oxide_is_under_the_film():
    ds = DataSet(name="test")
    ds.x_raw = np.linspace(0.01, 0.5, 50)
    ds.y_raw = np.ones_like(ds.x_raw)
    ds.error_raw = np.full_like(ds.x_raw, 0.1)
    ds.run_command()
    m = Model()
    m.data = DataList([ds])
    m.set_script(
        GenXScriptBuilder().build(
            {"f_d": 300.0, "f_sig": 4.0, "f_sld": 40.0, "s_len": 10.0, "beam_w": 0.1}
        )
    )
    m.compile_script()
    # GenX resolves a sample from the substrate up; the script's oxide starts at 15 A
    assert list(m.script_module.sample.resolveLayerParameters().d) == [0.0, 15.0, 300.0, 0.0]


def test_fit_recovers_a_film_on_buried_oxide():
    # Independent Parratt curve of air / film / SiO2 / Si, smeared with the script's Gaussian
    # resolution (sigma 0.002 1/A)
    q = np.linspace(0.05, 0.35, 300)
    qf = np.linspace(0.04, 0.36, 4000)
    w = np.exp(-0.5 * ((qf[None] - q[:, None]) / 0.002) ** 2)
    layers = [(0.0, 0.0, 0.0), (40.0, 200.0, 4.0), (SIO2, 20.0, 3.0), (SI, 0.0, 3.0)]
    r = (w @ parratt(qf, layers)) / w.sum(1)

    np.random.seed(0)
    guess = SimpleNamespace(thickness=206.0, roughness=4.12, sld=41.2)
    res = GenXFitter(q, r, guess, XRRConfig(sample_len_init=45.0, pop_size=20)).run(verbose=False)

    assert res["set_f_d"] == pytest.approx(200.0, rel=0.01)
    assert res["set_f_sig"] == pytest.approx(4.0, rel=0.1)
    assert res["set_f_sld"] == pytest.approx(40.0, rel=0.05)
    assert res["set_s_d"] == pytest.approx(20.0, abs=6.0)
    assert res["set_s_sld"] == pytest.approx(SIO2, abs=1.0)
