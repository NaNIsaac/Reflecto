# Reflecto

Single-layer X-ray reflectivity (XRR) analysis: a 1D CNN predicts film thickness, roughness and SLD from a measured curve (2θ, intensity `.dat`), then GenX refines a film-plus-SiO₂-on-Si model starting from that guess. Known problems: the GenX script puts the SiO₂ layer above the film while the training data had it below ([#3](https://github.com/NaNIsaac/Reflecto/issues/3)), and the background parameter is fitted but not applied ([#2](https://github.com/NaNIsaac/Reflecto/issues/2)).

The bundled weights (`src/reflecto/assets/best.pt`, `stats.pt`, `config.json`) come from experiment exp07 in [ReflectoExp](https://github.com/NaNIsaac/ReflectoExp) as of December 2025: `config.json` has the values of exp07's `config.py` at ReflectoExp b93c91f (2025-12-23: `exp_name` "cubic", fixed SiO₂ SLD 18.8, roughness 0-15 Å), not those of its current config.

Reflecto is one of the XRR research branches (CNN guess, then GenX differential evolution). [ReflectoLearn](https://github.com/NaNIsaac-Research/ReflectoLearn) carries an adapted copy of the GenX fit as `reflectolearn.genx_fit` (ReflectoLearn 6e160db). No comparison of the two approaches has been run, and the electron-density profile and density-ratio output here have no counterpart there.

## Usage

Python 3.13 or newer with [uv](https://docs.astral.sh/uv/). `pyproject.toml` has no build system, so the package is not installed and `src` goes on the path. From the repository root (PowerShell):

```powershell
uv sync
$env:PYTHONPATH = "src"
uv run main.py
```

`main.py` analyses `test_data/In2O3_Ar_30nm/#2_xrr.dat`, an In₂O₃ film of about 30 nm: CNN prediction, GenX fit, the density ratio against bulk and the figure of merit, then plots of the reflectivity curve and the electron-density profile. For another file, change `data_dir` and `file_path` in `main.py`.

```python
import reflecto
from reflecto.utils.math_utils import i0_normalize

df = reflecto.read_dat("sample.dat")  # two columns: tth, intensity (whitespace, # comments)
q = reflecto.tth2q(df["tth"].to_numpy())
R = i0_normalize(df["intensity"].to_numpy())
result = reflecto.analyze(q, R, fit=True)  # result["prediction"], result["fit_params"], result["fom"]
reflecto.plot_analysis_result(q, R, result)
```

## Structure

```
src/reflecto/
  core.py            analyze(): CNN inference and GenX fit
  engines/           CNN model, inference (inference.py), GenX fit (fitting.py, script_builder.py)
  simulate/          GenX reflectivity simulation, noise, SLD profiles
  utils/             .dat reading, 2θ to q, SLD and density conversion, GenX layer constants
  assets/            trained weights (best.pt, about 64 MB)
test_data/In2O3_Ar_30nm/   measured data, GenX .hgx projects, scripts for the result figures
```

Dependencies: numpy, torch, genx3, bumps, pandas, matplotlib, tqdm (`pyproject.toml`, `uv.lock`).
