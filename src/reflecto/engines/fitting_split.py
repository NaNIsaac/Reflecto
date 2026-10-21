"""
Split Mode Fitter: 하나의 필름을 여러 서브레이어로 분할하여 피팅
복잡한 밀도 프로파일이나 그라디언트 필름 분석에 유용
"""
import numpy as np
import time
from dataclasses import dataclass
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters
from .script_builder_multi import GenXScriptBuilderMultiLayer

@dataclass
class SplitFitConfig:
    """Split Mode 피팅 설정"""
    n_layers: int = 3              # 분할 레이어 수
    uniform_sld: bool = True       # True: SLD/Roughness 공유 (균일 필름)
    steps_initial: int = 300       # 초기 피팅
    steps_refine: int = 400        # 정제 피팅
    pop_size: int = 40
    sample_len_init: float = 10.0
    beam_width: float = 0.1

class SplitFitter:
    """
    Split Mode Fitter
    - 하나의 필름을 n_layers개로 분할
    - 각 레이어의 두께, SLD, 거칠기를 독립적으로 피팅
    - Gradient 필름이나 복잡한 구조 분석에 적합
    """
    
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: SplitFitConfig = None):
        self.q = q
        self.nn_params = nn_params
        self.config = config if config else SplitFitConfig()
        self.n_layers = self.config.n_layers

        # 데이터 정규화
        low_q_mask = q < 0.05
        norm_factor = np.max(R[low_q_mask]) if np.any(low_q_mask) else np.max(R[R > 0])
        if norm_factor <= 0:
            norm_factor = 1.0
            
        self.R = np.nan_to_num(R / norm_factor, nan=1e-12, posinf=1e-12)
        self.R = np.maximum(self.R, 1e-12)

        # SLD 변환
        self.nn_params.sld = float(self.nn_params.sld) * 28.1794

        self.model = self._initialize_model()
        self.pars_map = {}

    def _initialize_model(self) -> Model:
        ds = DataSet(name="Split_Fitter")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        init_params = {
            'film_thick': float(self.nn_params.thickness),
            'film_rough': float(self.nn_params.roughness),
            'film_sld': float(self.nn_params.sld),
            'sample_len': self.config.sample_len_init,
            'beam_width': self.config.beam_width,
        }
        
        builder = GenXScriptBuilderMultiLayer()
        script = builder.build(self.n_layers, init_params)
        model.set_script(script)
        model.compile_script()
        return model

    def _setup_parameters(self):
        pars = Parameters()
        total_thick = float(self.nn_params.thickness)
        sld = float(self.nn_params.sld)
        rough = float(self.nn_params.roughness)
        
        layer_thick = total_thick / self.n_layers

        def add_par(name, val, min_v, max_v, fit=True):
            p = pars.append(name, self.model)
            p.min, p.max = min_v, max_v
            p.value = np.clip(val, min_v, max_v)
            p.fit = fit
            key = name.replace("v.", "")
            self.pars_map[key] = p
            return p

        # 각 서브레이어 파라미터
        for i in range(self.n_layers):
            add_par(f"v.set_layer{i}_thick", layer_thick, 5.0, 500.0)
            add_par(f"v.set_layer{i}_sld", sld, 5.0, 150.0)
            add_par(f"v.set_layer{i}_rough", rough / 2, 0.0, 20.0)

        # SiO2
        add_par("v.set_oxide_thick", 10.0, 0.01, 30.0)
        add_par("v.set_oxide_sld", 18.0, 10.0, 25.0)
        add_par("v.set_oxide_rough", 3.0, 0.5, 15.0)

        # Instrument
        add_par("v.set_inst_i0", 1.0, 0.5, 2.0)
        add_par("v.set_inst_bkg", 1e-7, 1e-12, 1e-3)
        add_par("v.set_sample_len", self.config.sample_len_init, 5.0, 45.0)

        self.model.parameters = pars

    def _fit(self, method: str, steps: int, pop: int, fom_func) -> tuple[float, float]:
        start_time = time.time()
        self.model.set_fom_func(fom_func)
        result = self.model.bumps_fit(method=method, steps=steps, pop=pop)
        self.model.bumps_update_parameters(result)
        self.model.evaluate_sim_func()
        return self.model.fom, time.time() - start_time

    def _force_match_i0(self, verbose=True):
        self.model.evaluate_sim_func()
        sim_y = self.model.data[0].y_sim
        data_y = self.model.data[0].y
        
        max_sim = np.max(sim_y)
        max_data = np.max(data_y)
        
        if max_sim > 0:
            target_i0 = np.clip(max_data / max_sim, 0.5, 2.0)
            self.pars_map['set_inst_i0'].value = target_i0
            self.model.evaluate_sim_func()
            if verbose:
                print(f"I0 Fixed: {target_i0:.4f}")

    def _set_active_params(self, names: list[str]):
        for p in self.model.parameters:
            p.fit = False
        for name in names:
            if name in self.pars_map:
                self.pars_map[name].fit = True

    def _estimate_thickness_from_fft(self, verbose=True):
        """
        FFT를 사용하여 프린지 주기에서 두께를 추정
        q^4로 프레넬 감쇠를 보정하여 평탄화 후 FFT
        프린지 간격 ΔQ ≈ 2π/d → d ≈ 2π/ΔQ
        """
        # q^4 보정: 프레넬 감쇠 제거
        q_safe = np.maximum(self.q, 0.01)  # 0 근처 방지
        R_flat = self.R * (q_safe ** 4)
        
        # 로그 스케일 (선택적, 진폭 균일화)
        # log_R = np.log10(np.maximum(R_flat, 1e-20))
        
        # 평균 제거 (DC 성분 제거)
        R_centered = R_flat - np.mean(R_flat)
        
        # Q 간격
        dq = np.mean(np.diff(self.q))
        n = len(self.q)
        
        # FFT 수행
        fft_vals = np.abs(np.fft.rfft(R_centered))
        freqs = np.fft.rfftfreq(n, dq)
        
        # 낮은 주파수(DC) 제외하고 피크 찾기
        # 두께 50Å ~ 2000Å에 해당하는 주파수 범위
        # d = 2π/ΔQ, ΔQ = 1/freq → freq = d/(2π)
        min_freq = 50 / (2 * np.pi)   # 50Å에 해당
        max_freq = 2000 / (2 * np.pi)  # 2000Å에 해당
        
        valid_mask = (freqs > min_freq) & (freqs < max_freq)
        if not np.any(valid_mask):
            if verbose: print("FFT: No valid frequency range for thickness")
            return None
        
        valid_fft = fft_vals.copy()
        valid_fft[~valid_mask] = 0
        
        peak_idx = np.argmax(valid_fft)
        peak_freq = freqs[peak_idx]
        
        if peak_freq <= 0:
            if verbose: print("FFT: No valid peak")
            return None
        
        # 두께 추정: d = 2π * peak_freq
        estimated_d = 2 * np.pi * peak_freq
        
        if verbose:
            print(f"FFT Thickness (q⁴ corrected): {estimated_d:.1f} Å")
        
        return estimated_d

    def run(self, verbose=True):
        if verbose:
            print("\n" + "="*80)
            print(f"Split Mode Fitter ({self.n_layers} layers)")
            print("="*80)

        start_total = time.time()
        self._setup_parameters()
        self.model.evaluate_sim_func()
        
        self._force_match_i0(verbose=verbose)
        
        # FFT로 프린지에서 두께 추정
        fft_thick = self._estimate_thickness_from_fft(verbose=verbose)
        if fft_thick and fft_thick > 30:
            # FFT 두께로 레이어 초기화 (프린지 위치 보존)
            layer_thick = fft_thick / self.n_layers
            for i in range(self.n_layers):
                self.pars_map[f'set_layer{i}_thick'].value = layer_thick
            if verbose:
                print(f"  -> Layer init: {layer_thick:.1f} Å × {self.n_layers}")
            self.model.evaluate_sim_func()
        
        # Step 1: 두께 피팅 (uniform_sld 모드: SLD/Roughness는 Layer 0만)
        if self.config.uniform_sld:
            if verbose: print(f"\n[Step 1] Uniform Mode: Thick per layer + Shared SLD...")
            # 두께만 모든 레이어, SLD/Roughness는 Layer 0만
            layer_params = [f"set_layer{i}_thick" for i in range(self.n_layers)]
            layer_params.extend(["set_layer0_sld", "set_layer0_rough"])
        else:
            if verbose: print(f"\n[Step 1] Fitting {self.n_layers} sub-layers (Thick + SLD)...")
            layer_params = []
            for i in range(self.n_layers):
                layer_params.extend([f"set_layer{i}_thick", f"set_layer{i}_sld"])
        
        self._set_active_params(layer_params)
        fom1, t1 = self._fit("de", self.config.steps_initial, self.config.pop_size, fom_funcs.logR1)
        
        # uniform_sld 모드: Layer 0의 SLD/Roughness를 다른 레이어에 동기화
        if self.config.uniform_sld:
            sld0 = self.pars_map['set_layer0_sld'].value
            rough0 = self.pars_map['set_layer0_rough'].value
            for i in range(1, self.n_layers):
                self.pars_map[f'set_layer{i}_sld'].value = sld0
                self.pars_map[f'set_layer{i}_rough'].value = rough0
            self.model.evaluate_sim_func()
        
        if verbose: self._print_layers(f"LAYERS ({t1:.1f}s)")
        
        # Step 2: 거칠기 포함 전체 피팅 (uniform_sld 모드에서는 bounds 동기화)
        if verbose: print(f"\n[Step 2] Full refinement with roughness...")
        
        if self.config.uniform_sld:
            # Layer 0 기준으로 다른 레이어의 SLD/Roughness bounds를 ±5%로 제한
            sld0 = self.pars_map['set_layer0_sld'].value
            rough0 = self.pars_map['set_layer0_rough'].value
            
            for i in range(self.n_layers):
                p_sld = self.pars_map[f'set_layer{i}_sld']
                p_rough = self.pars_map[f'set_layer{i}_rough']
                
                # SLD: ±5% 범위로 제한
                p_sld.min = max(5.0, sld0 * 0.95)
                p_sld.max = min(150.0, sld0 * 1.05)
                p_sld.value = sld0
                
                # Roughness: ±10% 범위로 제한
                p_rough.min = max(0.0, rough0 * 0.9)
                p_rough.max = min(20.0, rough0 * 1.1)
                p_rough.value = rough0
            
            if verbose:
                print(f"  -> SLD bounds locked: {sld0*0.95:.1f} ~ {sld0*1.05:.1f} (±5%)")
        
        all_params = [k for k in self.pars_map.keys() if k != "set_inst_i0"]
        self._set_active_params(all_params)
        fom2, t2 = self._fit("de", self.config.steps_refine, self.config.pop_size, fom_funcs.logR1)
        if verbose: self._print_layers(f"REFINE ({t2:.1f}s)")
        
        # Step 3: Dream polish
        if verbose: print(f"\n[Step 3] Dream polish...")
        fom3, t3 = self._fit("dream", 200, self.config.pop_size, fom_funcs.logR1)
        
        if verbose:
            self._print_layers(f"FINAL ({t3:.1f}s)")
            print(f"\nTotal: {time.time() - start_total:.1f}s, FOM: {self.model.fom:.4f}")
            print("="*80)

        return self._collect_results()

    def _print_layers(self, title: str):
        print(f"\n{title}")
        print("-" * 60)
        print(f"{'Layer':<10} {'Thick (Å)':<12} {'SLD':<10} {'Rough (Å)':<10}")
        print("-" * 60)
        
        total_thick = 0
        for i in range(self.n_layers):
            t = self.pars_map[f'set_layer{i}_thick'].value
            s = self.pars_map[f'set_layer{i}_sld'].value
            r = self.pars_map[f'set_layer{i}_rough'].value
            total_thick += t
            print(f"Layer {i:<4} {t:<12.2f} {s:<10.1f} {r:<10.2f}")
        
        print("-" * 60)
        print(f"Total: {total_thick:.1f} Å, FOM: {self.model.fom:.4f}")

    def _collect_results(self) -> dict:
        self.model.evaluate_sim_func()
        
        layers = []
        total_thick = 0
        for i in range(self.n_layers):
            t = self.pars_map[f'set_layer{i}_thick'].value
            s = self.pars_map[f'set_layer{i}_sld'].value
            r = self.pars_map[f'set_layer{i}_rough'].value
            total_thick += t
            layers.append({'thick': t, 'sld': s, 'rough': r})
        
        return {
            'layers': layers,
            'total_thickness': total_thick,
            'R_sim': self.model.data[0].y_sim,
            'fom': self.model.fom
        }
