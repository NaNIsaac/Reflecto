# Reflecto

Single-layer X-ray reflectivity (XRR) analysis: a 1D CNN predicts film thickness, roughness and SLD from a measured curve, then GenX refines the fit.

XRR 측정 곡선(2θ, intensity `.dat`)에서 학습된 CNN 으로 박막 두께 · 거칠기 · SLD 초기값을 추정하고, GenX(genx3) 로 막/SiO₂/Si 모델을 정밀 피팅하는 라이브러리. 개인 연구용 실험 코드이며 유지보수되는 패키지는 아님.

모델 가중치(`src/reflecto/assets/best.pt`, `stats.pt`, `config.json`)는 [ReflectoExp](https://github.com/NaNIsaac/ReflectoExp) 의 exp07 학습 결과다 (`config.json` 의 `base_dir`). 이후 XRR 역문제 작업은 [ReflectoLearn](https://github.com/NaNIsaac/ReflectoLearn) 에서 진행.

## Requirements

Python 3.13+, numpy, torch, genx3, bumps, pandas, matplotlib, tqdm (`pyproject.toml`, `uv.lock`).

## Usage

```powershell
uv sync
$env:PYTHONPATH = "src"     # pyproject 에 build-system 이 없어 패키지가 설치되지 않음
uv run main.py              # test_data/In2O3_Ar_30nm/#2_xrr.dat 분석 (저장소 루트에서 실행)
```

`main.py` 는 In₂O₃ 30 nm 시료로 AI 예측 → GenX 피팅 → 벌크 밀도 대비 비율 · FOM 출력 → 반사율 곡선과 전자밀도 프로파일을 그린다. 다른 파일은 `main.py` 의 `data_dir` / `file_path` 를 수정.

```python
import reflecto
from reflecto.utils.math_utils import i0_normalize

df = reflecto.read_dat("sample.dat")            # 2열: tth, intensity (공백 구분, # 주석)
q = reflecto.tth2q(df["tth"].to_numpy())
R = i0_normalize(df["intensity"].to_numpy())
result = reflecto.analyze(q, R, fit=True)       # result["prediction"], result["fit_params"], result["fom"]
reflecto.plot_analysis_result(q, R, result)
```

## Structure

```
src/reflecto/
  core.py            analyze(): CNN 추론 + GenX 피팅
  engines/           CNN 모델 · 추론(inference.py) · GenX 피팅(fitting.py, script_builder.py)
  simulate/          GenX 반사율 시뮬레이션, 노이즈, SLD 프로파일
  utils/             .dat 읽기, 2θ→q, SLD ↔ 밀도 변환, GenX 층 상수
  assets/            학습된 가중치 (best.pt ≈ 64 MB)
test_data/In2O3_Ar_30nm/   측정 데이터, GenX .hgx, 결과 그림용 스크립트
```
