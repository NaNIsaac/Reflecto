class GenXScriptBuilderMultiLayer:
    """다중 레이어를 지원하는 GenX 스크립트 빌더 (Split Mode)"""
    
    def build(self, n_layers: int, init_params: dict) -> str:
        """
        n_layers: 필름을 몇 개의 레이어로 분할할지
        init_params: 초기값 dict (film_thick, film_sld, film_rough 등)
        """
        
        # 각 서브레이어의 초기값 계산
        total_thick = init_params['film_thick']
        layer_thick = total_thick / n_layers
        sld = init_params['film_sld']
        rough = init_params['film_rough']
        
        # 동적으로 레이어 변수 생성
        layer_vars = []
        for i in range(n_layers):
            layer_vars.append(f"""
        # Layer {i}
        self.layer{i}_thick = {layer_thick}
        self.layer{i}_sld   = {sld}
        self.layer{i}_rough = {rough / 2 if i < n_layers - 1 else rough}""")
        
        # Setter/Getter 생성
        setters_getters = []
        for i in range(n_layers):
            setters_getters.append(f"""
    def set_layer{i}_thick(self, v): self.layer{i}_thick = float(v)
    def get_layer{i}_thick(self): return self.layer{i}_thick
    def set_layer{i}_sld(self, v): self.layer{i}_sld = float(v)
    def get_layer{i}_sld(self): return self.layer{i}_sld
    def set_layer{i}_rough(self, v): self.layer{i}_rough = float(v)
    def get_layer{i}_rough(self): return self.layer{i}_rough""")
        
        # Layer 객체 생성 코드
        layer_defs = []
        for i in range(n_layers):
            layer_defs.append(f"Layer{i} = Layer(d=v.layer{i}_thick, sigma=v.layer{i}_rough, f=to_f(v.layer{i}_sld), dens=1)")
        
        # Stack 구성 (위에서 아래로: Layer0 = 표면, LayerN-1 = 기판 가까이)
        layer_names = [f"Layer{i}" for i in range(n_layers)]
        stack_layers = ", ".join(layer_names + ["SiO2"])
        
        # Sim 함수 내 동적 업데이트 코드
        sim_updates = []
        for i in range(n_layers):
            sim_updates.append(f"    Layer{i}.d, Layer{i}.sigma, Layer{i}.f = v.layer{i}_thick, v.layer{i}_rough, to_f(v.layer{i}_sld)")
        
        script = f"""
import numpy as np
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

def to_f(sld_val):
    return complex((sld_val * 1e-6) / r_e, 0)

class Sim_Vars:
    def __init__(self):
        {"".join(layer_vars)}
        
        # SiO2 (Fixed/Guess)
        self.oxide_thick = 15.0
        self.oxide_rough = 3.0
        self.oxide_sld   = 18.8

        # Instrument
        self.inst_i0    = 1.0
        self.sample_len = {init_params['sample_len']}
        self.beam_width = {init_params['beam_width']}
        self.inst_bkg   = 1e-7

    # === Auto-generated setters/getters ===
    {"".join(setters_getters)}
    
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
{chr(10).join(layer_defs)}
SiO2 = Layer(d=v.oxide_thick, sigma=v.oxide_rough, f=to_f(v.oxide_sld), dens=1)

sample = Sample(Stacks=[Stack(Layers=[{stack_layers}])], Ambient=Amb, Substrate=Sub)

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
{chr(10).join(sim_updates)}
    SiO2.d, SiO2.sigma, SiO2.f = v.oxide_thick, v.oxide_rough, to_f(v.oxide_sld)
    inst.I0, inst.samplelen, inst.beamw = v.inst_i0, v.sample_len, v.beam_width
    return [Specular(d.x, sample, inst) for d in data]
"""
        return script
