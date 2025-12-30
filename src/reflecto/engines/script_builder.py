class GenXScriptBuilder:
    """GenX 모델 내부에서 실행될 파이썬 스크립트를 생성합니다."""
    
    def build(self, init_params: dict[str, float]) -> str:
        return f"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

def to_f(sld_val):
    return complex((sld_val * 1e-6) / r_e, 0)

class Sim_Vars:
    def __init__(self):
        # Film (AI Guess)
        self.film_thick = {init_params['film_thick']}
        self.film_rough = {init_params['film_rough']}
        self.film_sld   = {init_params['film_sld']}

        # SiO2 (Fixed/Guess)
        self.oxide_thick = 15.0
        self.oxide_rough = 3.0
        self.oxide_sld   = 18.8

        # Instrument
        self.inst_i0    = 1.0
        self.sample_len = {init_params['sample_len']}
        self.beam_width = {init_params['beam_width']}
        self.inst_bkg   = 1e-7

    # =================================================================
    # [FIX] GenX requires BOTH Setters AND Getters for every parameter
    # =================================================================

    # --- Film ---
    def set_film_thick(self, v): self.film_thick = float(v)
    def get_film_thick(self): return self.film_thick

    def set_film_rough(self, v): self.film_rough = float(v)
    def get_film_rough(self): return self.film_rough

    def set_film_sld(self, v): self.film_sld = float(v)
    def get_film_sld(self): return self.film_sld
    
    # --- SiO2 ---
    def set_oxide_thick(self, v): self.oxide_thick = float(v)
    def get_oxide_thick(self): return self.oxide_thick

    def set_oxide_rough(self, v): self.oxide_rough = float(v)
    def get_oxide_rough(self): return self.oxide_rough

    def set_oxide_sld(self, v): self.oxide_sld = float(v)
    def get_oxide_sld(self): return self.oxide_sld
    
    # --- Instrument ---
    def set_inst_i0(self, v): self.inst_i0 = float(v)
    def get_inst_i0(self): return self.inst_i0

    def set_sample_len(self, v): self.sample_len = float(v)
    def get_sample_len(self): return self.sample_len

    def set_beam_width(self, v): self.beam_width = float(v)
    def get_beam_width(self): return self.beam_width

    def set_inst_bkg(self, v): self.inst_bkg = float(v)
    def get_inst_bkg(self): return self.inst_bkg

v = Sim_Vars()

# Layer Definitions
Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=to_f(20.07), dens=1, sigma=3.0)
Film = Layer(d=v.film_thick, sigma=v.film_rough, f=to_f(v.film_sld), dens=1)
SiO2 = Layer(d=v.oxide_thick, sigma=v.oxide_rough, f=to_f(v.oxide_sld), dens=1)

sample = Sample(Stacks=[Stack(Layers=[Film, SiO2])], Ambient=Amb, Substrate=Sub)

# Instrument Definitions
inst = Instrument(
    probe=Probe.xray, wavelength=1.54, coords=Coords.q,
    I0=v.inst_i0, Ibkg=v.inst_bkg, res=0.002,
    restype=ResType.fast_conv,
    footype=FootType.gauss,
    samplelen=v.sample_len,
    beamw=v.beam_width
)

def Sim(data):
    # Dynamic update hook
    Film.d, Film.sigma, Film.f = v.film_thick, v.film_rough, to_f(v.film_sld)
    SiO2.d, SiO2.sigma, SiO2.f = v.oxide_thick, v.oxide_rough, to_f(v.oxide_sld)
    inst.I0, inst.samplelen, inst.beamw = v.inst_i0, v.sample_len, v.beam_width
    return [Specular(d.x, sample, inst) for d in data]
"""