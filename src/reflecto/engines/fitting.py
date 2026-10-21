import numpy as np
import time
from dataclasses import dataclass
from genx import fom_funcs
from genx.data import DataList, DataSet
from genx.model import Model
from genx.parameters import Parameters
from .script_builder import GenXScriptBuilder

@dataclass
class XRRConfig:
    """범용 XRR 분석을 위한 물리학적 파라미터 설정"""
    wavelength: float = 1.5406
    beam_width: float = 0.1
    sample_len_init: float = 10.0

    # 피팅 단계별 steps 설정 (간소화된 2단계 전략)
    steps_film: int = 400      # 1단계: 필름 파라미터 (Thick, Rough, SLD)
    steps_final: int = 600     # 2단계: 전체 동시 피팅

    pop_size: int = 50         # 적당한 탐색력

class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: XRRConfig = None):
        self.q = q
        self.nn_params = nn_params
        self.config = config if config else XRRConfig()

        # === 데이터 정규화 ===
        low_q_mask = q < 0.05
        if np.any(low_q_mask):
            norm_factor = np.max(R[low_q_mask])
        else:
            norm_factor = np.max(R[R > 0])

        if norm_factor <= 0:
            norm_factor = 1.0
            
        self.R = np.nan_to_num(R / norm_factor, nan=1e-12, posinf=1e-12)
        self.R = np.maximum(self.R, 1e-12)

        # === Electron Density → SLD 변환 ===
        self.nn_params.sld = float(self.nn_params.sld) * 28.1794

        self.model = self._initialize_genx_model()
        self.pars_map = {}

    def _initialize_genx_model(self) -> Model:
        ds = DataSet(name="Reflecto_Fitter")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        init_vals = {
            'film_thick': float(self.nn_params.thickness),
            'film_rough': float(self.nn_params.roughness),
            'film_sld': float(self.nn_params.sld),
            'sample_len': self.config.sample_len_init,
            'beam_width': self.config.beam_width,
            'inst_i0': 1.0,
            'inst_bkg': 1e-7
        }
        builder = GenXScriptBuilder()
        model.set_script(builder.build(init_vals))
        model.compile_script()
        return model

    def _setup_parameters(self):
        pars = Parameters()
        d_ai = float(self.nn_params.thickness)
        sld_ai = float(self.nn_params.sld)
        sig_ai = float(self.nn_params.roughness)

        def add_par(name, val, min_v, max_v, fit=True):
            p = pars.append(name, self.model)
            p.min, p.max = min_v, max_v
            p.value = np.clip(val, min_v, max_v)
            p.fit = fit
            self.pars_map[name.replace("v.set_", "set_").replace("v.", "")] = p
            return p

        # [1] Main Film: 넓은 범위
        add_par("v.set_film_thick", d_ai, 30.0, 2000.0)
        add_par("v.set_film_rough", sig_ai, 0.0, 30.0)
        add_par("v.set_film_sld", sld_ai, 5.0, 150.0)

        # [2] Substrate Oxide
        add_par("v.set_oxide_thick", 10.0, 0.01, 30.0)
        add_par("v.set_oxide_rough", 3.0, 0.5, 15.0)
        add_par("v.set_oxide_sld", 18.0, 10.0, 25.0)

        # [3] Instrument
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
        elapsed = time.time() - start_time
        return self.model.fom, elapsed

    def _force_match_i0(self, verbose=True):
        """I0를 데이터 기반으로 강제 설정"""
        self.model.evaluate_sim_func()
        
        sim_y = self.model.data[0].y_sim
        data_y = self.model.data[0].y
        
        max_sim = np.max(sim_y)
        max_data = np.max(data_y)
        
        if max_sim > 0:
            target_i0 = max_data / max_sim
            self.pars_map['set_inst_i0'].value = np.clip(target_i0, 0.5, 2.0)
            self.model.evaluate_sim_func()
            
            if verbose:
                print(f"I0 Fixed: {target_i0:.4f}")

    def _match_critical_angle(self, verbose=True):
        """Critical Angle에서 SLD 추정"""
        data_y = self.model.data[0].y
        
        q_limit_mask = (self.q < 0.1) & (self.q > 0.01)
        if not np.any(q_limit_mask):
            return

        roi_y = data_y[q_limit_mask]
        roi_q = self.q[q_limit_mask]
        
        mask_below_half = roi_y < 0.5
        if not np.any(mask_below_half):
            return
            
        if mask_below_half[0]:
            dy = np.diff(roi_y)
            idx = np.argmin(dy)
            qc_est = roi_q[idx]
        else:
            idx = np.argmax(mask_below_half)
            qc_est = roi_q[idx]
            
        target_sld = (qc_est**2 / (16 * np.pi)) * 1e6
        
        if verbose:
            print(f"Critical Angle: Qc~{qc_est:.4f} -> SLD {target_sld:.1f}")

        if target_sld > 25.0:
            p = self.pars_map['set_film_sld']
            p.value = float(np.clip(target_sld, p.min, p.max))

    def _set_active_params(self, names: list[str]):
        for p in self.model.parameters:
            p.fit = False
        for name in names:
            if name in self.pars_map:
                self.pars_map[name].fit = True

    def run(self, verbose=True):
        if verbose:
            self._print_header()

        start_total = time.time()
        self._setup_parameters()
        self.model.evaluate_sim_func()
        
        # === Step 0: I0 고정 & Qc에서 SLD 추정 ===
        self._match_critical_angle(verbose=verbose)
        self._force_match_i0(verbose=verbose)
        
        # === Step 1: 필름 3 파라미터 피팅 (I0 고정) ===
        if verbose: print("\n[Step 1] Film Parameters (Thick, Rough, SLD)...")
        self._set_active_params(["set_film_thick", "set_film_rough", "set_film_sld"])
        fom1, t1 = self._fit("de", self.config.steps_film, self.config.pop_size, fom_funcs.logR1)
        if verbose: self._print_status("FILM", t1)
        
        # === Step 2: 전체 동시 피팅 ===
        if verbose: print("\n[Step 2] Full Simultaneous Fit...")
        all_params = [k for k in self.pars_map.keys() if k != "set_inst_i0"]
        self._set_active_params(all_params)
        fom2, t2 = self._fit("de", self.config.steps_final, self.config.pop_size, fom_funcs.logR1)
        if verbose: self._print_status("FULL", t2)
        
        # === Step 3: Dream Polish ===
        if verbose: print("\n[Step 3] Dream Polish...")
        fom3, t3 = self._fit("dream", 300, self.config.pop_size, fom_funcs.logR1)
        if verbose: 
            self._print_status("DREAM", t3)
            print(f"\nTotal: {time.time() - start_total:.1f}s, Final FOM: {self.model.fom:.3g}")
            print("="*100)

        return self._collect_results()

    def _print_header(self):
        print("\n" + "="*100)
        print("Reflecto XRR Fitting")
        print("="*100)
        self._fmt = "{:<10} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}"
        print(self._fmt.format("Step", "FOM", "Thick", "Rough", "SLD", "SiO2_d", "SiO2_sld", "I0", "Time"))
        print("-" * 100)

    def _print_status(self, step_name: str, elapsed: float):
        p = self.pars_map
        fom_val = self.model.fom
        print(self._fmt.format(
            step_name,
            f"{fom_val:.4f}",
            f"{p['set_film_thick'].value:.1f}",
            f"{p['set_film_rough'].value:.2f}",
            f"{p['set_film_sld'].value:.1f}",
            f"{p['set_oxide_thick'].value:.1f}",
            f"{p['set_oxide_sld'].value:.1f}",
            f"{p['set_inst_i0'].value:.3f}",
            f"{elapsed:.1f}s"
        ))

    def _collect_results(self) -> dict:
        self.model.evaluate_sim_func()
        results = {name: p.value for name, p in self.pars_map.items()}
        results.update({'R_sim': self.model.data[0].y_sim, 'fom': self.model.fom})
        return results