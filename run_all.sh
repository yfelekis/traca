#!/usr/bin/env bash
# ==========================================================================
# run_all.sh — Production experiment runs (14-config default)
#
# Default configs (paper figures):
#   1. ATE gaussian subfamily symmetric          (cross 6x6)
#   2. ATE gaussian subfamily directional d=0.5  (cross 6x6)
#   3. ATCE gaussian subfamily symmetric         (zip 6-pt)
#   4. LiLuCaS empirical ew subfamily symmetric  (zip 6-pt)
#   5. LiLuCaS empirical ew subfamily dir d=0.5  (zip 6-pt)
#   6. LiLuCaS gaussian ew subfamily symmetric   (zip 6-pt)
#   7. Portland env-only Q-restricted             (6-pt eps sweep)
#   8-14. ATE misspec sweep d in {-0.5,0,0.3,0.4,0.6,0.7,1.0} + gate (zip 6-pt)
#
# Additional (not in default, pass by name):
#   atce_full         ATCE full-joint variant
#   lilucas_full      LiLuCaS empirical full-joint variants (sym + dir)
#
# Usage:
#   bash run_all.sh              # run everything needed for paper figures
#   bash run_all.sh ate          # run only ATE (sym + dir)
#   bash run_all.sh atce         # run ATCE subfamily
#   bash run_all.sh atce_full    # run ATCE full-joint (consistency check)
#   bash run_all.sh lilucas_full # run LiLuCaS full-joint (consistency check)
# ==========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Training grid
EPS_ETA_6="0.0 0.2 0.5 1.0 2.0 4.0"

# Eval grid
RHO_TEST="0.0 0.2 0.5 1.0 2.0 4.0"
EVAL_K=100
EVAL_SEED=2026

OUT_ROOT="results_production"


# --------------------------------------------------------------------------
# ATE — Gaussian, EntrywiseBox, 6×6 cross grid (sym + dir)
# --------------------------------------------------------------------------
run_ate() {
    echo ""
    echo "=================================================================="
    echo "  ATE — 2 configs (sym + dir) × 6×6 cross grid, 5 folds"
    echo "=================================================================="

    ATE_CONFIGS=(
        "gaussian_entrywise_subfamily:ate_gaussian_entrywise_subfamily:ate_sym"
        "gaussian_entrywise_subfamily_directional:ate_gaussian_entrywise_subfamily_directional:ate_dir"
    )

    for entry in "${ATE_CONFIGS[@]}"; do
        IFS=':' read -r yaml_stem exp_name out_tag <<< "$entry"
        echo ""
        echo "--- ATE ${out_tag} ---"

        python traca_train.py \
            --config "configs/ate/${yaml_stem}.yaml" \
            --eps_values $EPS_ETA_6 \
            --eta_values $EPS_ETA_6 \
            --grid_mode cross \
            --k_folds 5 \
            --out_dir "$OUT_ROOT/${out_tag}" \
            --progress

        python traca_run_evaluation.py \
            --results "$OUT_ROOT/${out_tag}/${exp_name}/traca_cv_results.pkl" \
            --target_config data_configs/ate_target.yaml
    done
}


# --------------------------------------------------------------------------
# ATCE — Gaussian, EntrywiseBox, 6-point diagonal zip (sub + full)
# --------------------------------------------------------------------------
run_atce() {
    echo ""
    echo "=================================================================="
    echo "  ATCE — subfamily × 6-pt zip, 5 folds"
    echo "=================================================================="

    echo ""
    echo "--- ATCE subfamily ---"

    python traca_train.py \
        --config "configs/atce/gaussian_z_entrywise_subfamily.yaml" \
        --eps_values $EPS_ETA_6 \
        --eta_values $EPS_ETA_6 \
        --grid_mode zip \
        --k_folds 5 \
        --out_dir "$OUT_ROOT/atce_subfamily" \
        --progress

    python traca_radius_eval.py \
        --results "$OUT_ROOT/atce_subfamily/atce_gaussian_z_entrywise_subfamily/traca_cv_results.pkl" \
        --rho_test $RHO_TEST --K $EVAL_K --seed $EVAL_SEED
}

run_atce_full() {
    echo ""
    echo "=================================================================="
    echo "  ATCE full — full-joint variant (consistency check, not in paper)"
    echo "=================================================================="

    python traca_train.py \
        --config "configs/atce/gaussian_z_entrywise_full.yaml" \
        --eps_values $EPS_ETA_6 \
        --eta_values $EPS_ETA_6 \
        --grid_mode zip \
        --k_folds 5 \
        --out_dir "$OUT_ROOT/atce_full" \
        --progress

    python traca_radius_eval.py \
        --results "$OUT_ROOT/atce_full/atce_gaussian_z_entrywise_full/traca_cv_results.pkl" \
        --rho_test $RHO_TEST --K $EVAL_K --seed $EVAL_SEED
}


# --------------------------------------------------------------------------
# LiLuCaS empirical — EntrywiseBox, 4 configs (sym + dir × sub + full)
# --------------------------------------------------------------------------
run_lilucas() {
    echo ""
    echo "=================================================================="
    echo "  LiLuCaS empirical — 2 configs (sub sym + sub dir) × 6-pt zip, 5 folds"
    echo "=================================================================="

    LILUCAS_CONFIGS=(
        "light_empirical_entrywise_subfamily:lilucas_light_entrywise_subfamily:lilucas_ew_sub"
        "light_empirical_entrywise_subfamily_directional:lilucas_light_entrywise_subfamily_directional:lilucas_dir_sub"
    )

    for entry in "${LILUCAS_CONFIGS[@]}"; do
        IFS=':' read -r yaml_stem exp_name out_tag <<< "$entry"
        echo ""
        echo "--- LiLuCaS ${out_tag} ---"

        python traca_train.py \
            --config "configs/lilucas/${yaml_stem}.yaml" \
            --eps_values $EPS_ETA_6 \
            --eta_values $EPS_ETA_6 \
            --grid_mode zip \
            --k_folds 5 \
            --out_dir "$OUT_ROOT/${out_tag}" \
            --progress

        python traca_radius_eval.py \
            --results "$OUT_ROOT/${out_tag}/${exp_name}/traca_cv_results.pkl" \
            --rho_test $RHO_TEST --K $EVAL_K --seed $EVAL_SEED
    done
}

run_lilucas_full() {
    echo ""
    echo "=================================================================="
    echo "  LiLuCaS empirical full — full-joint variants (consistency check, not in paper)"
    echo "=================================================================="

    LILUCAS_FULL_CONFIGS=(
        "light_empirical_entrywise_full:lilucas_light_entrywise_full:lilucas_ew_full"
        "light_empirical_entrywise_full_directional:lilucas_light_entrywise_full_directional:lilucas_dir_full"
    )

    for entry in "${LILUCAS_FULL_CONFIGS[@]}"; do
        IFS=':' read -r yaml_stem exp_name out_tag <<< "$entry"
        echo ""
        echo "--- LiLuCaS ${out_tag} (full) ---"

        python traca_train.py \
            --config "configs/lilucas/${yaml_stem}.yaml" \
            --eps_values $EPS_ETA_6 \
            --eta_values $EPS_ETA_6 \
            --grid_mode zip \
            --k_folds 5 \
            --out_dir "$OUT_ROOT/${out_tag}" \
            --progress

        python traca_radius_eval.py \
            --results "$OUT_ROOT/${out_tag}/${exp_name}/traca_cv_results.pkl" \
            --rho_test $RHO_TEST --K $EVAL_K --seed $EVAL_SEED
    done
}


# --------------------------------------------------------------------------
# LiLuCaS Gaussian — EntrywiseBox, 1 config (subfamily only, symmetric)
# --------------------------------------------------------------------------
run_lilucas_gaussian() {
    echo ""
    echo "=================================================================="
    echo "  LiLuCaS Gaussian — 1 config (subfamily) × 6-pt zip, 5 folds"
    echo "=================================================================="

    LILUCAS_GAU_CONFIGS=(
        "light_gaussian_entrywise_subfamily:lilucas_light_gaussian_entrywise_subfamily:lilucas_gew_sub"
    )

    for entry in "${LILUCAS_GAU_CONFIGS[@]}"; do
        IFS=':' read -r yaml_stem exp_name out_tag <<< "$entry"
        echo ""
        echo "--- LiLuCaS Gaussian ${out_tag} ---"

        python traca_train.py \
            --config "configs/lilucas/${yaml_stem}.yaml" \
            --eps_values $EPS_ETA_6 \
            --eta_values $EPS_ETA_6 \
            --grid_mode zip \
            --k_folds 5 \
            --out_dir "$OUT_ROOT/${out_tag}" \
            --progress

        python traca_radius_eval.py \
            --results "$OUT_ROOT/${out_tag}/${exp_name}/traca_cv_results.pkl" \
            --rho_test $RHO_TEST --K $EVAL_K --seed $EVAL_SEED
    done
}


# --------------------------------------------------------------------------
# ATE misspecification sweep — 7+1 deltas, 6-point zip, 5 folds
# --------------------------------------------------------------------------
run_ate_misspec() {
    echo ""
    echo "=================================================================="
    echo "  ATE misspec — 7 configs x 6-pt zip, 5 folds + delta=0.5 gate"
    echo "=================================================================="

    MISSPEC_ENTRIES=(
        "delta_-0.5:ate_misspec_dm0.5"
        "delta_0.0:ate_misspec_d0.0"
        "delta_0.3:ate_misspec_d0.3"
        "delta_0.4:ate_misspec_d0.4"
        "delta_0.6:ate_misspec_d0.6"
        "delta_0.7:ate_misspec_d0.7"
        "delta_1.0:ate_misspec_d1.0"
    )

    for entry in "${MISSPEC_ENTRIES[@]}"; do
        IFS=':' read -r yaml_stem exp_name <<< "$entry"
        echo ""
        echo "--- ATE misspec ${yaml_stem} ---"

        python traca_train.py \
            --config "configs/ate/misspec/${yaml_stem}.yaml" \
            --eps_values $EPS_ETA_6 \
            --eta_values $EPS_ETA_6 \
            --grid_mode zip \
            --k_folds 5 \
            --out_dir "$OUT_ROOT/ate_misspec" \
            --progress

        python traca_run_evaluation.py \
            --results "$OUT_ROOT/ate_misspec/${exp_name}/traca_cv_results.pkl" \
            --target_config data_configs/ate_target.yaml
    done

    # delta=0.5: reuse the directional config for the correctness gate
    echo ""
    echo "--- ATE misspec delta=0.5 (directional config, for gate) ---"

    python traca_train.py \
        --config configs/ate/gaussian_entrywise_subfamily_directional.yaml \
        --eps_values $EPS_ETA_6 \
        --eta_values $EPS_ETA_6 \
        --grid_mode zip \
        --k_folds 5 \
        --out_dir "$OUT_ROOT/ate_misspec" \
        --progress

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/ate_misspec/ate_gaussian_entrywise_subfamily_directional/traca_cv_results.pkl" \
        --target_config data_configs/ate_target.yaml

    # Correctness gate
    python traca_figures/verify_misspec_sweep.py \
        --correctness_gate_only \
        --results_dir "$OUT_ROOT/ate_misspec" \
        --published_dir "$OUT_ROOT"
}


# --------------------------------------------------------------------------
# Portland
# --------------------------------------------------------------------------
run_portland() {
    echo ""
    echo "=================================================================="
    echo "  Portland — 6-pt ε sweep, 5 folds, real Fanno Creek target"
    echo "=================================================================="

    python traca_train.py \
        --config experiments/configs/portland_backdoor_gaussian_qrestricted.yaml \
        --sweep_axis eps \
        --values $EPS_ETA_6 \
        --k_folds 5 \
        --out_dir "$OUT_ROOT/portland" \
        --progress

    python traca_run_evaluation.py \
        --results "$OUT_ROOT/portland/portland_backdoor_gaussian_qrestricted/traca_cv_results.pkl"
}


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
ALL_EXPERIMENTS="ate atce lilucas lilucas_gaussian portland ate_misspec"

usage() {
    echo "Usage: bash run_all.sh [experiment ...]"
    echo "  Available: $ALL_EXPERIMENTS"
    echo "  Consistency checks (not in default): atce_full lilucas_full"
    echo "  No arguments -> run all configs needed for paper figures."
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
        atce_full)        run_atce_full ;;
        lilucas)          run_lilucas ;;
        lilucas_full)     run_lilucas_full ;;
        lilucas_gaussian) run_lilucas_gaussian ;;
        ate_misspec)      run_ate_misspec ;;
        portland)         run_portland ;;
        *)                echo "ERROR: unknown experiment '$exp'"; usage ;;
    esac
done

echo ""
echo "=================================================================="
echo "  ALL DONE"
echo "=================================================================="
