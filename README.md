# TraCA

## Repository structure

```
traca/                     Core library (ambiguity, certificates, losses, optimiser)
lan_scm.py                 LAN SCM module — data generation, interventions, CLI
build_portland.py           Portland real-world benchmark builder
data_configs/               SCM specifications 
configs/                    Experiment configs: atce/ (4), lilucas/ (8)
  ate/                      ATE configs (2)
  ate/misspec/              ATE misspecification sweep (7)
experiments/                Experiment framework (run.py, evaluate.py, portland_eval.py)
  configs/                  Portland configs (3)
shift_grids/                Compound shift grids for ATCE and LiLUCAS evaluation
traca_figures/              Figure generators and sweep verification
tests/                      Test suite (15 modules)
run_all.sh                  Full production pipeline
run_smoke.sh                Fast smoke test (~2 min)
```

## Reproducing results

### 1. Install

```bash
pip install -e '.[all]'
```

### 2. Generate data

```bash
# Synthetic benchmarks
python lan_scm.py data_configs/atce.yaml
python lan_scm.py data_configs/ate.yaml
python lan_scm.py data_configs/lilucas_light.yaml

# Portland real-world benchmark (requires network access;
# downloads from EDI and USGS ScienceBase, ~30 s on first run;
# subsequent runs use cached files in data/portland/raw/)
python build_portland.py
```

### 3. Train and evaluate

```bash
# Full production run (all benchmarks needed for paper figures)
bash run_all.sh

# Or run individual benchmarks
bash run_all.sh ate
bash run_all.sh atce
bash run_all.sh lilucas
bash run_all.sh lilucas_gaussian
bash run_all.sh portland
bash run_all.sh ate_misspec

# Full-joint variants (consistency check, not reported in the paper)
bash run_all.sh atce_full
bash run_all.sh lilucas_full
```

### 4. Generate figures

```bash
python traca_figures/generate_final.py
python traca_figures/generate_linear_variants.py
python traca_figures/generate_misspec_sweep.py
```

## Pipeline scripts

| Script | Role |
|--------|------|
| `traca_train.py` | K-fold CV training per (ε, η) grid point |
| `traca_run_evaluation.py` | Evaluation against real or synthetic targets |
| `traca_radius_eval.py` | Radius-sampling evaluation (no retraining) |
| `traca_figures/verify_misspec_sweep.py` | Correctness gate for ATE misspecification sweep |

## Running tests

```bash
pytest tests/ -q
```

## Dependencies

Core (in `pyproject.toml`): `numpy>=1.24`, `scipy>=1.10`, `pyyaml>=6.0`, `joblib>=1.3`

Optional (`pip install -e '.[all]'`): `torch` (autograd gradients), `matplotlib`, `seaborn` (figures), `tqdm` (progress bars), `pandas` (evaluation CSVs)
