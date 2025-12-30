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

    # 피팅 단계별 steps 설정 (low-q 우선 전략)
    steps_i0: int = 200      # I0 맞춤: 짧고 정밀하게
    steps_edge: int = 400    # Critical edge: SLD, 두께, 거칠기
    steps_substrate: int = 500  # 기판 + 전체 진동
    steps_fine: int = 600    # Dream 정밀 탐색

    pop_size: int = 30

    # FOM 임계값 설정 (모두 동일한 기준)
    fom_threshold_initial: float = 0.05  # 초기 허용 오차
    fom_threshold_after_step: float = 0.03  # 단계 후 허용 오차
    max_bound_expansions: int = 3  # 최대 반복 확장 횟수

class GenXFitter:
    def __init__(self, q: np.ndarray, R: np.ndarray, nn_params, config: XRRConfig = None):
        self.q = q
        self.nn_params = nn_params
        self.config = config if config else XRRConfig()
        self.expansion_count = 0

        # === 데이터 정규화: 전반사 영역 기준으로 정확히 정규화 ===
        # low-q 영역 (q < 0.05)의 최대값으로 정규화
        low_q_mask = q < 0.05
        if np.any(low_q_mask):
            norm_factor = np.max(R[low_q_mask])
        else:
            norm_factor = np.max(R[R > 0])

        if norm_factor <= 0:
            norm_factor = 1.0
            
        self.R = np.nan_to_num(R / norm_factor, nan=1e-12, posinf=1e-12)

        # === 신경망 출력: Electron Density → SLD 변환 ===
        # 신경망이 출력한 값은 electron density (e/Å³)
        # SLD = electron_density * r_e * 1e6 (r_e = 2.81794e-13 cm)
        # 간단히: SLD ≈ electron_density * 28.1794
        self.nn_params.sld = float(self.nn_params.sld) * 28.1794

        self.model = self._initialize_genx_model()
        self.pars_map = {}

    def _initialize_genx_model(self) -> Model:
        ds = DataSet(name="Reflecto_Ultimate_Fitter")
        ds.x_raw, ds.y_raw = self.q, self.R
        ds.error_raw = np.maximum(self.R * 0.1, 1e-9)
        ds.run_command()

        model = Model()
        model.data = DataList([ds])

        # 초기값 설정: 정규화된 데이터이므로 I0는 1.0 근처
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

        # [1] Main Film: 초기 범위를 넓게
        add_par("v.set_film_thick", d_ai, max(5.0, d_ai*0.2), min(2000.0, d_ai*5.0))
        add_par("v.set_film_rough", sig_ai, 0.0, min(30.0, sig_ai*5.0))
        add_par("v.set_film_sld", sld_ai, max(1.0, sld_ai*0.2), min(200.0, sld_ai*5.0))

        # [2] Substrate Oxide (SiO2)
        add_par("v.set_oxide_thick", 10.0, 0.01, 20.0)
        add_par("v.set_oxide_rough", 3.0, 0.5, 15.0)
        add_par("v.set_oxide_sld", 13.0, 5.0, 20.0)

        # [3] Instrument: I0 범위를 매우 좁게!
        add_par("v.set_inst_i0", 1.0, 0.5, 2.0)
        add_par("v.set_inst_bkg", 1e-7, 1e-12, 1e-3)
        add_par("v.set_sample_len", self.config.sample_len_init, 5.0, 45.0) 

        self.model.parameters = pars

    def _fit_and_evaluate(self, method: str, steps: int, pop: int, fom_func) -> tuple[float, float]:
        """피팅 실행 및 (FOM, 경과시간) 반환"""
        start_time = time.time()
        self.model.set_fom_func(fom_func)
        result = self.model.bumps_fit(method=method, steps=steps, pop=pop)
        self.model.bumps_update_parameters(result)
        self.model.evaluate_sim_func()
        elapsed = time.time() - start_time
        return self.model.fom, elapsed

    def _should_expand_bounds(self, current_fom: float, threshold: float) -> bool:
        """FOM 기준으로 범위 확장이 필요한지 판단"""
        return current_fom > threshold and self.expansion_count < self.config.max_bound_expansions

    def _expand_bounds_iteratively(self):
        """실제 파라미터 범위 확장 (FOM 체크 없음)"""
        self.expansion_count += 1
        print(f"[WARN] Expanding bounds (try {self.expansion_count}/{self.config.max_bound_expansions})")

        p_map = self.pars_map
        # 메인 필름 파라미터만 확장 (계측기 파라미터는 고정)
        d_current = p_map['set_film_thick'].value
        p_map['set_film_thick'].min = max(5.0, d_current * 0.05)  # 5%로 확장
        p_map['set_film_thick'].max = min(2000.0, d_current * 4.0)  # 400%로 확장

        sig_current = p_map['set_film_rough'].value
        p_map['set_film_rough'].min = 0.0
        p_map['set_film_rough'].max = min(30.0, sig_current * 3.0)

        sld_current = p_map['set_film_sld'].value
        p_map['set_film_sld'].min = max(1.0, sld_current * 0.05)
        p_map['set_film_sld'].max = min(200.0, sld_current * 4.0)

    def _force_match_i0(self, verbose=True):
        """데이터와 시뮬레이션의 최대값 비율을 계산하여 I0를 강제 설정"""
        self.model.evaluate_sim_func()
        
        sim_y = self.model.data[0].y_sim
        data_y = self.model.data[0].y
        
        max_sim = np.max(sim_y)
        max_data = np.max(data_y)
        
        if max_sim > 0:
            current_i0 = self.pars_map['set_inst_i0'].value
            # 목표 I0 = 현재 I0 * (데이터 최대값 / 시뮬레이션 최대값)
            target_i0 = current_i0 * (max_data / max_sim)
            
            # 파라미터 업데이트
            self.pars_map['set_inst_i0'].value = target_i0
            self.model.evaluate_sim_func() # 변경된 I0로 다시 계산
            
            if verbose:
                print(f"Forced I0 adjustment: {current_i0:.4f} -> {target_i0:.4f} "
                              f"(Data Max: {max_data:.2e} / Sim Max: {max_sim:.2e})")

    def _run_step_with_retries(self, name: str, active_params: list[str], steps: int, retry_steps: int, fom_func, verbose=True):
        """공통 피팅 단계 실행 (피팅 -> 필요시 범위확장 -> 재피팅)"""
        self._set_active_params(active_params)
        fom_val, elapsed = self._fit_and_evaluate("de", steps, self.config.pop_size if name != "I0" else 10, fom_func)
        
        if self._should_expand_bounds(fom_val, self.config.fom_threshold_after_step):
            self._expand_bounds_iteratively()
            fom_re, elapsed_re = self._fit_and_evaluate("de", retry_steps, self.config.pop_size if name != "I0" else 10, fom_func)
            elapsed += elapsed_re
            if verbose:
                self._print_status(f"{name}+", elapsed_re)
        else:
            if verbose:
                self._print_status(name, elapsed)
        
        return elapsed

    def _match_critical_angle(self, verbose=True):
        """데이터에서 Critical Angle(Qc)을 추정하여 Substrate SLD 설정"""
        data_y = self.model.data[0].y
        
        # 정규화된 데이터이므로 0.5 지점을 찾음 (간단한 Qc 추정)
        # 노이즈를 피하기 위해 앞부분(Q<0.1)에서만 검색
        q_limit_mask = self.q < 0.1
        roi_y = data_y[q_limit_mask]
        roi_q = self.q[q_limit_mask]
        
        # 값이 0.5 이하로 떨어지는 첫 지점 찾기
        mask_below_half = roi_y < 0.5
        if np.any(mask_below_half):
            idx = np.argmax(mask_below_half) # True가 처음 나오는 인덱스
            qc_est = roi_q[idx]
            
            # SLD 계산: SLD = (Qc^2 / 16pi) * 1e6
            target_sld = (qc_est**2 / (16 * np.pi)) * 1e6
            
            # 파라미터 업데이트
            old_val = self.pars_map['set_oxide_sld'].value
            # 범위 제한
            target_sld = np.clip(target_sld, self.pars_map['set_oxide_sld'].min, self.pars_map['set_oxide_sld'].max)
            
            self.pars_map['set_oxide_sld'].value = target_sld
            
            if verbose:
                print(f"Critical Angle Match: Qc~{qc_est:.4f} -> SLD {old_val:.2f} -> {target_sld:.2f}")
        else:
            if verbose:
                print("Could not detect Critical Angle (all > 0.5 in low Q). Skipping SLD adjustment.")

    def run(self, verbose=True):
        if verbose:
            self._print_header()

        start_total = time.time()
        self._setup_parameters()

        # === Step 0: 초기 평가 ===
        self.model.evaluate_sim_func()
        init_fom = self.model.fom
        
        # 1. Critical Angle 맞춤
        self._match_critical_angle(verbose=verbose)

        # 2. I0 강제 맞춤 (New I0)
        self._force_match_i0(verbose=verbose)

        if verbose:
            print(f"Initial FOM: {init_fom:.3g} -> Adjusted FOM: {self.model.fom:.3g}\n")
        
        # === Step 1: Background & Sample Length (I0 제외!) ===
        # 이전에는 I0도 맞췄으나, 이제는 강제 설정된 I0를 고정하고 나머지만 맞춤.
        self._run_step_with_retries(
            "I0(Fix)", 
            ["set_inst_bkg", "set_sample_len"], 
            self.config.steps_i0, 
            100, 
            fom_funcs.R1, 
            verbose=verbose
        )
        
        # === Step 2: Critical Edge (SLD, 두께, 거칠기) ===
        self._run_step_with_retries(
            "EDGE", 
            ["set_film_thick", "set_film_sld", "set_film_rough"], 
            self.config.steps_edge, 
            200, 
            fom_funcs.logR1, 
            verbose=verbose
        )
        
        # === Step 3: Substrate & Oscillations ===
        self._run_step_with_retries(
            "SUB", 
            ["set_film_thick", "set_film_rough", "set_oxide_thick", "set_oxide_sld", "set_oxide_rough"], 
            self.config.steps_substrate, 
            300, 
            fom_funcs.logR2, 
            verbose=verbose
        )
        
        # === Step 4: Final Polish (Dream) ===
        # I0를 제외한 모든 파라미터 활성화
        all_params = [k for k in self.pars_map.keys() if k != "set_inst_i0"]
        self._set_active_params(all_params)
        
        fom_dream, elapsed_dream = self._fit_and_evaluate("dream", self.config.steps_fine, self.config.pop_size, fom_funcs.logR2)

        if verbose:
            self._print_status("DREAM", elapsed_dream)

        # === 최종 출력 ===
        if verbose:
            self._print_status("FINAL", time.time() - start_total)
            print(f"\nTotal: {time.time() - start_total:.1f}s, "
                f"Expansions: {self.expansion_count}, Final FOM: {self.model.fom:.3g}\n")
            print("="*120 + "\n")

        return self._collect_results()
    

    def _set_active_params(self, active_names: list[str]):
        """파라미터 활성화/비활성화"""
        for p in self.model.parameters:
            p.fit = False
        for name in active_names:
            if name in self.pars_map:
                self.pars_map[name].fit = True

    def _print_header(self):
        print("\nReflecto XRR Fitting Progress\n")
        # Fixed width formatting
        self._fmt = "{:<12} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8} {:<8} {:<8} {:<8}"
        print("="*130)
        print(self._fmt.format("Step", "FOM (log)", "Thick", "Rough", "SLD", "SiO2_d", "SiO2_s", "SiO2_sld", "I0", "L", "Bkg", "Time(s)"))
        print("-" * 130)

    def _print_status(self, step_name: str, elapsed: float):
        """상태 출력 (FOM은 항상 현재 model.fom 사용)"""
        p = self.pars_map
        fom_val = self.model.fom
        log_fom = np.log10(fom_val) if fom_val > 0 else -np.inf
        b_val = np.log10(max(1e-12, p['set_inst_bkg'].value))
        fom_str = f"{fom_val:5.3g} ({log_fom:5.2f})"

        print(self._fmt.format(
            step_name,
            fom_str,
            f"{p['set_film_thick'].value:.2f}",
            f"{p['set_film_rough'].value:.2f}",
            f"{p['set_film_sld'].value:.2f}",
            f"{p['set_oxide_thick'].value:.2f}",
            f"{p['set_oxide_rough'].value:.2f}",
            f"{p['set_oxide_sld'].value:.2f}",
            f"{p['set_inst_i0'].value:.2f}",
            f"{p['set_sample_len'].value:.1f}",
            f"{b_val:.1f}",
            f"{elapsed:.2f}"
        ))

    def _collect_results(self) -> dict:
        self.model.evaluate_sim_func()
        results = {name: p.value for name, p in self.pars_map.items()}
        results.update({'R_sim': self.model.data[0].y_sim, 'fom': self.model.fom})
        return results