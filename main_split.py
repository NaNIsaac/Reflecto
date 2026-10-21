"""
Split Mode 테스트 스크립트
Usage: uv run main_split.py [n_layers]
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import reflecto
from reflecto.engines.fitting_split import SplitFitter, SplitFitConfig

def main():
    # 인자로 레이어 수 받기 (기본값 3)
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    print(f"🚀 Split Mode Test (n_layers={n_layers})")
    print("="*60)
    
    # 데이터 로드
    data_dir = Path("test_data") / "KBJ"
    file_path = data_dir / "ALD_E_no_XRR.dat"
    
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return
    
    print(f"[1] Loading: {file_path}")
    df = reflecto.read_dat(file_path)
    tth = df['tth'].values
    R = df['intensity'].values
    qs = reflecto.tth2q(tth, wavelen=1.5406)

    # AI 추론
    print("[2] AI Inference...")
    engine = reflecto.load_default_engine()
    ai_result = engine.predict(qs, R)

    # 원본 AI SLD 저장 (fitter가 변환하기 전에)
    ai_sld_raw = ai_result.sld  # electron density
    ai_sld_converted = ai_sld_raw * 28.18  # SLD로 변환

    print(f"    AI Guess: Thick={ai_result.thickness:.1f}Å, SLD={ai_sld_converted:.1f}, Rough={ai_result.roughness:.2f}Å")
    
    # AI Curve 생성
    from reflecto.simulate.simulate import param2refl
    ai_curve = param2refl(qs, [ai_result], sio2_param=None)
    
    # Split Mode 피팅
    print(f"\n[3] Split Mode Fitting ({n_layers} layers)...")
    config = SplitFitConfig(n_layers=n_layers, uniform_sld=True, steps_initial=300, steps_refine=400)
    fitter = SplitFitter(qs, R, ai_result, config)
    result = fitter.run(verbose=True)
    
    # 평균값 계산
    avg_sld = np.mean([l['sld'] for l in result['layers']])
    avg_rough = np.mean([l['rough'] for l in result['layers']])
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 COMPARISON")
    print("="*60)
    print(f"{'':12} {'AI Guess':>15} {'Split Fit':>15}")
    print("-"*45)
    print(f"{'Thickness':<12} {ai_result.thickness:>12.1f} Å {result['total_thickness']:>12.1f} Å")
    print(f"{'SLD':<12} {ai_sld_converted:>15.1f} {avg_sld:>15.1f}")
    print(f"{'Roughness':<12} {ai_result.roughness:>12.2f} Å {avg_rough:>12.2f} Å")
    print(f"{'FOM':<12} {'N/A':>15} {result['fom']:>15.4f}")
    print("="*60)
    
    # 정규화
    R_norm = R / np.max(R)
    
    # 플롯 (반사율 비교만)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(qs, R_norm, 'ko', ms=2, label='Measured', alpha=0.7)
    ax.semilogy(qs, ai_curve, 'b--', lw=1.5, label=f'AI Guess ({ai_result.thickness:.0f}Å)')
    ax.semilogy(qs, result['R_sim'], 'r-', lw=2, label=f'Split Fit ({result["total_thickness"]:.0f}Å, SLD={avg_sld:.1f})')
    
    ax.set_xlabel(r'$q$ [Å$^{-1}$]', fontsize=12)
    ax.set_ylabel('Reflectivity', fontsize=12)
    ax.set_title(f'XRR Split Fit ({n_layers} layers, FOM={result["fom"]:.4f})', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
