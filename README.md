# Reflecto

Superseded by [ReflectoLearn](https://github.com/NaNIsaac/ReflectoLearn); the GenX fit lives on as `reflectolearn.genx_fit` (`genx` extra).

Single-layer X-ray reflectivity (XRR) analysis: a 1D CNN predicts film thickness, roughness and SLD from a measured curve (2θ, intensity `.dat`), then GenX refines a film / SiO₂ / Si model starting from that guess.

The bundled weights (`src/reflecto/assets/best.pt`, `stats.pt`, `config.json`) come from experiment exp07 in [ReflectoExp](https://github.com/NaNIsaac/ReflectoExp).

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
