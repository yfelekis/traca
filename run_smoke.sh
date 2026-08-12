#!/usr/bin/env bash
# ==========================================================================
# run_smoke.sh — Smoke test mirror of run_all.sh
#
# Same structure, selection mechanism, and function names as run_all.sh but
# scaled down: n_iters=15, 2 folds, 2 grid points, 2 shifts.
# Verifies plumbing (paths, flags, JSON refs, output dirs) end-to-end in
# ~2 minutes so you know run_all.sh will work before waiting on compute.
#
# n_iters can't be overridden via CLI, so each function patches the config
# to a tmpdir with n_iters=15 before running.
#
# Usage:
#   bash run_smoke.sh              # smoke everything
#   bash run_smoke.sh ate portland # smoke only ATE and Portland
# ==========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SMOKE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/traca_smoke_XXXXXX")
echo "[smoke] temp dir: $SMOKE_DIR"

# Reduced grids
SMOKE_EPS_ETA="0.1 0.5"
SMOKE_PORTLAND_EPS="0.2 1.0"
SMOKE_FOLDS=2
OUT_ROOT="$SMOKE_DIR/results"

# 2-point shift grids (subset of full 25-point grids)
SMOKE_ATCE_GRID='[
  [{"shift_type":"noise_mean","magnitude":0.1,"node":0},{"shift_type":"mechanism_edge","magnitude":0.05,"edge":[0,2]},{"shift_type":"mechanism_edge","magnitude":0.05,"edge":[1,2]}],
  [{"shift_type":"noise_mean","magnitude":1.0,"node":0},{"shift_type":"mechanism_edge","magnitude":0.5,"edge":[0,2]},{"shift_type":"mechanism_edge","magnitude":0.5,"edge":[1,2]}]
]'

SMOKE_LILUCAS_GRID='[
  [{"shift_type":"noise_mean","magnitude":0.1,"node":3},{"shift_type":"mechanism_edge","magnitude":0.1,"edge":[0,3]},{"shift_type":"mechanism_edge","magnitude":0.1,"edge":[1,3]}],
  [{"shift_type":"noise_mean","magnitude":1.0,"node":3},{"shift_type":"mechanism_edge","magnitude":0.9,"edge":[0,3]},{"shift_type":"mechanism_edge","magnitude":0.9,"edge":[1,3]}]
]'

# Helper: patch n_iters in a config YAML → tmpdir
patch_config() {
    local src="$1"
    local dst="$SMOKE_DIR/$(basename "$src")"
    python -c "
import yaml, sys
with open('$src') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('training', {})['n_iters'] = 15
with open('$dst', 'w') as f:
    yaml.dump(cfg, f)
print('$dst')
"
}


# --------------------------------------------------------------------------
run_ate() {
    echo ""
    echo "=================================================================="
    echo "  [SMOKE] ATE — 2×2 cross, 2 folds, real target"
    echo "=================================================================="

    local cfg; cfg=$(patch_config configs/ate/gaussian_entrywise_subfamily.yaml)

    python traca_train.py \
        --config "$cfg" \
        --eps_values $SMOKE_EPS_ETA \
        --eta_values $SMOKE_EPS_ETA \
        --grid_mode cross \
        --k_folds $SMOKE_FOLDS \
        --out_dir "$OUT_ROOT/ate"

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/ate/ate_gaussian_entrywise_subfamily/traca_cv_results.pkl" \
        --target_config data_configs/ate_target.yaml

    echo "[smoke] ATE: OK"
}


# --------------------------------------------------------------------------
run_atce() {
    echo ""
    echo "=================================================================="
    echo "  [SMOKE] ATCE — 1 config (full), 2-pt zip, 2 folds, 2 shifts"
    echo "=================================================================="

    local cfg; cfg=$(patch_config configs/atce/gaussian_z_entrywise_full.yaml)

    python traca_train.py \
        --config "$cfg" \
        --eps_values $SMOKE_EPS_ETA \
        --eta_values $SMOKE_EPS_ETA \
        --grid_mode zip \
        --k_folds $SMOKE_FOLDS \
        --out_dir "$OUT_ROOT/atce_full"

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/atce_full/atce_gaussian_z_entrywise_full/traca_cv_results.pkl" \
        --shift_grid_json "$SMOKE_ATCE_GRID"

    echo "[smoke] ATCE: OK"
}


# --------------------------------------------------------------------------
run_lilucas() {
    echo ""
    echo "=================================================================="
    echo "  [SMOKE] LiLuCaS — 1 config (frob full), 2-pt zip, 2 folds, 2 shifts"
    echo "=================================================================="

    local cfg; cfg=$(patch_config configs/lilucas/light_empirical_frobenius_full.yaml)

    python traca_train.py \
        --config "$cfg" \
        --eps_values $SMOKE_EPS_ETA \
        --eta_values $SMOKE_EPS_ETA \
        --grid_mode zip \
        --k_folds $SMOKE_FOLDS \
        --out_dir "$OUT_ROOT/lilucas_frob_full"

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/lilucas_frob_full/lilucas_light_frobenius_full/traca_cv_results.pkl" \
        --shift_grid_json "$SMOKE_LILUCAS_GRID"

    echo "[smoke] LiLuCaS: OK"
}


# --------------------------------------------------------------------------
run_lilucas_gaussian() {
    echo ""
    echo "=================================================================="
    echo "  [SMOKE] LiLuCaS Gaussian — 1 config (ew sub), 2-pt zip, 2 folds"
    echo "=================================================================="

    local cfg; cfg=$(patch_config configs/lilucas/light_gaussian_entrywise_subfamily.yaml)

    python traca_train.py \
        --config "$cfg" \
        --eps_values $SMOKE_EPS_ETA \
        --eta_values $SMOKE_EPS_ETA \
        --grid_mode zip \
        --k_folds $SMOKE_FOLDS \
        --out_dir "$OUT_ROOT/lilucas_gew_sub"

    python traca_radius_eval.py \
        --results "$OUT_ROOT/lilucas_gew_sub/lilucas_light_gaussian_entrywise_subfamily/traca_cv_results.pkl" \
        --rho_test $SMOKE_EPS_ETA --K 2 --seed 2026

    echo "[smoke] LiLuCaS Gaussian: OK"
}


# --------------------------------------------------------------------------
run_portland() {
    echo ""
    echo "=================================================================="
    echo "  [SMOKE] Portland — 2-pt ε, 2 folds, auto-discovered target"
    echo "=================================================================="

    local cfg; cfg=$(patch_config experiments/configs/portland_backdoor_gaussian_qrestricted.yaml)

    python traca_train.py \
        --config "$cfg" \
        --sweep_axis eps \
        --values $SMOKE_PORTLAND_EPS \
        --k_folds $SMOKE_FOLDS \
        --out_dir "$OUT_ROOT/portland"

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/portland/portland_backdoor_gaussian_qrestricted/traca_cv_results.pkl"

    echo "[smoke] Portland: OK"
}


# --------------------------------------------------------------------------
# Dispatcher (identical to run_all.sh)
# --------------------------------------------------------------------------
ALL_EXPERIMENTS="ate atce lilucas lilucas_gaussian portland"

usage() {
    echo "Usage: bash run_smoke.sh [experiment ...]"
    echo "  Available experiments: $ALL_EXPERIMENTS"
    echo "  No arguments → smoke all."
    exit 1
}

if [ $# -eq 0 ]; then
    experiments="$ALL_EXPERIMENTS"
else
    experiments="$*"
fi

for exp in $experiments; do
    case "$exp" in
        ate)              run_ate ;;
        atce)             run_atce ;;
        lilucas)          run_lilucas ;;
        lilucas_gaussian) run_lilucas_gaussian ;;
        portland)         run_portland ;;
        *)                echo "ERROR: unknown experiment '$exp'"; usage ;;
    esac
done

echo ""
echo "=================================================================="
echo "  [SMOKE] ALL DONE — output in $SMOKE_DIR"
echo "  (safe to delete: rm -rf $SMOKE_DIR)"
echo "=================================================================="
