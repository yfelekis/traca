"""Structural geometry checks for experiment configs.

P1 (Column rule): the mechanism ambiguity's nonzero entries must be a subset of
    {incoming edges of the shifted nodes}.  A node v's mechanism = column v of W.
    If the adversary can perturb entry (i, j), that perturbation changes node j's
    mechanism — so j must be a shifted node with parents.

P2 (Root-node rule): a shifted root node has no parents, hence no incoming edges,
    hence no mechanism to shift.  The mechanism budget for column j of a shifted
    root j must be identically zero.

These tests would have caught the pre-2026-07-14 ATCE bug (mechanism budget on
W[0,1] = X's mechanism instead of column 2 = Y's mechanism) and the Portland bug
(eta=0.3 and shifted_nodes=[2,3] instead of eta=0 and shifted_nodes=[0]).
"""

import numpy as np
import pytest
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def _W_from_data_config(data_cfg):
    """Build W matrix from a data_config's edge list."""
    scm = data_cfg["scm"]
    var_names = scm["var_names"]
    d = len(var_names)
    idx = {n: i for i, n in enumerate(var_names)}
    W = np.zeros((d, d))
    for src, dst, w in scm["edges"]:
        W[idx[src], idx[dst]] = w
    return W, var_names


# Portland DAG (no data_config YAML — built from build_portland.py)
# d=4, var_names=[Z, S, X, Y]
# Edges: Z->X (0,2), Z->Y (0,3), S->Y (1,3), X->Y (2,3)
PORTLAND_W = np.array([
    [0, 0, 1, 1],  # Z→X, Z→Y (weights don't matter, only structure)
    [0, 0, 0, 1],  # S→Y
    [0, 0, 0, 1],  # X→Y
    [0, 0, 0, 0],
], dtype=float)
PORTLAND_VAR_NAMES = ["Z", "S", "X", "Y"]


def _resolve_W(cfg):
    """Get W matrix for a config, from data_config or hardcoded Portland."""
    data = cfg.get("data", {})
    if "source" in data:
        dc = _load_yaml(data["source"])
        return _W_from_data_config(dc)
    # Portland — uses bundle_path, not a data_config
    if "bundle_path" in data:
        return PORTLAND_W.copy(), PORTLAND_VAR_NAMES
    raise ValueError(f"Cannot resolve DAG for config: {data}")


def _root_nodes(W):
    """Root node indices: no parents → column is all-zero."""
    d = W.shape[0]
    return {j for j in range(d) if np.allclose(W[:, j], 0)}


def _incoming_edges_of(W, nodes):
    """Union of incoming edges {(i, j)} for all j in nodes."""
    d = W.shape[0]
    edges = set()
    for j in nodes:
        for i in range(d):
            if i < j and W[i, j] != 0:  # strict upper-triangular
                edges.add((i, j))
    return edges


def _live_mechanism_entries(mech_cfg, d):
    """Set of (i, j) where the mechanism adversary has nonzero budget."""
    typ = mech_cfg["type"]
    shifted_rows = mech_cfg.get("shifted_rows", [])

    if typ == "EntrywiseBox":
        B = np.array(mech_cfg["B"])
        return {
            (i, j)
            for i in shifted_rows
            for j in range(i + 1, d)
            if B[i, j] > 0
        }

    if typ == "FrobeniusBall":
        eta = mech_cfg.get("eta", 0.0)
        if eta == 0.0:
            return set()
        mask = mech_cfg.get("entry_mask")
        return {
            (i, j)
            for i in shifted_rows
            for j in range(i + 1, d)
            if mask is None or mask[i][j] > 0
        }

    raise ValueError(f"Unknown mechanism type: {typ}")


# ---------------------------------------------------------------------------
# Paper benchmark configs (excludes tau=I baselines and non-paper configs)
# ---------------------------------------------------------------------------

PAPER_BENCHMARKS = [
    # ATE
    "configs/ate/gaussian_entrywise_subfamily.yaml",
    "configs/ate/gaussian_entrywise_subfamily_directional.yaml",
    # ATCE (entrywise — paper benchmarks)
    "configs/atce/gaussian_z_entrywise_full.yaml",
    "configs/atce/gaussian_z_entrywise_subfamily.yaml",
    # LiLuCaS (all 4 paper variants)
    "configs/lilucas/light_empirical_frobenius_full.yaml",
    "configs/lilucas/light_empirical_frobenius_subfamily.yaml",
    "configs/lilucas/light_empirical_entrywise_full.yaml",
    "configs/lilucas/light_empirical_entrywise_subfamily.yaml",
    "configs/lilucas/light_empirical_entrywise_full_directional.yaml",
    "configs/lilucas/light_empirical_entrywise_subfamily_directional.yaml",
    # LiLuCaS Gaussian EntrywiseBox
    "configs/lilucas/light_gaussian_entrywise_full.yaml",
    "configs/lilucas/light_gaussian_entrywise_subfamily.yaml",
    # Portland (all Gaussian variants)
    "experiments/configs/portland_backdoor_gaussian_qrestricted.yaml",
    "experiments/configs/portland_backdoor_gaussian_eps0_2.yaml",
    # Portland empirical
    "experiments/configs/portland_backdoor.yaml",
]

# tau=I baselines: mechanism geometry should still be correct (column rule),
# but shifted_nodes=[] so P1 is checked differently.
TAU_I_BASELINES = [
    "configs/atce/gaussian_z_frobenius_full.yaml",
    "configs/atce/gaussian_z_frobenius_single.yaml",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestColumnRule:
    """P1: nonzero mechanism entries ⊆ incoming edges of shifted nodes."""

    @pytest.mark.parametrize("config_path", PAPER_BENCHMARKS,
                             ids=[Path(p).stem for p in PAPER_BENCHMARKS])
    def test_paper_benchmark(self, config_path):
        cfg = _load_yaml(config_path)
        W, var_names = _resolve_W(cfg)
        d = W.shape[0]

        shifted_nodes = set(cfg["constructive_class"]["shifted_nodes"])
        assert len(shifted_nodes) > 0, "Paper benchmark must have shifted_nodes"

        mech_cfg = cfg["ambiguity"]["mechanism"]
        live = _live_mechanism_entries(mech_cfg, d)
        allowed = _incoming_edges_of(W, shifted_nodes)

        extra = live - allowed
        assert extra == set(), (
            f"Mechanism entries {extra} are NOT incoming edges of "
            f"shifted_nodes={shifted_nodes} (var_names={var_names}). "
            f"Allowed={allowed}, Live={live}"
        )

    @pytest.mark.parametrize("config_path", TAU_I_BASELINES,
                             ids=[Path(p).stem for p in TAU_I_BASELINES])
    def test_tau_I_baseline_column_aligned(self, config_path):
        """tau=I baselines have shifted_nodes=[] but mechanism budget should
        still target columns of nodes whose mechanism plausibly shifts
        (not arbitrary columns)."""
        cfg = _load_yaml(config_path)
        W, var_names = _resolve_W(cfg)
        d = W.shape[0]

        mech_cfg = cfg["ambiguity"]["mechanism"]
        live = _live_mechanism_entries(mech_cfg, d)

        # All live entries should target columns that are actual DAG edges
        for i, j in live:
            assert W[i, j] != 0, (
                f"Mechanism entry ({i},{j}) is not a DAG edge "
                f"(W[{i},{j}]=0, var_names={var_names})"
            )

        # All live entries should target the same column set (single mechanism)
        if live:
            cols = {j for _, j in live}
            # Verify every targeted column has at least one parent
            roots = _root_nodes(W)
            root_cols = cols & roots
            assert root_cols == set(), (
                f"tau=I baseline targets root-node columns {root_cols} "
                f"(root nodes have no mechanism to shift)"
            )


class TestRootNodeRule:
    """P2: shifted root nodes must have zero mechanism budget in their column."""

    @pytest.mark.parametrize("config_path", PAPER_BENCHMARKS + TAU_I_BASELINES,
                             ids=[Path(p).stem for p in
                                  PAPER_BENCHMARKS + TAU_I_BASELINES])
    def test_shifted_root_has_no_mechanism_budget(self, config_path):
        cfg = _load_yaml(config_path)
        W, var_names = _resolve_W(cfg)
        d = W.shape[0]

        shifted_nodes = set(cfg["constructive_class"]["shifted_nodes"])
        roots = _root_nodes(W)

        mech_cfg = cfg["ambiguity"]["mechanism"]
        live = _live_mechanism_entries(mech_cfg, d)
        live_cols = {j for _, j in live}

        shifted_roots = shifted_nodes & roots
        violated_roots = shifted_roots & live_cols

        assert violated_roots == set(), (
            f"Shifted root nodes {violated_roots} have nonzero mechanism "
            f"budget — root nodes have no incoming edges, so their mechanism "
            f"cannot shift. (var_names={var_names}, "
            f"roots={roots}, shifted_nodes={shifted_nodes})"
        )

    @pytest.mark.parametrize("config_path", PAPER_BENCHMARKS + TAU_I_BASELINES,
                             ids=[Path(p).stem for p in
                                  PAPER_BENCHMARKS + TAU_I_BASELINES])
    def test_all_roots_shifted_implies_eta_zero(self, config_path):
        """If ALL shifted nodes are roots, FrobeniusBall eta must be 0
        (the entire mechanism budget is vacuous)."""
        cfg = _load_yaml(config_path)
        W, _ = _resolve_W(cfg)

        shifted_nodes = set(cfg["constructive_class"]["shifted_nodes"])
        roots = _root_nodes(W)

        mech_cfg = cfg["ambiguity"]["mechanism"]
        if mech_cfg["type"] != "FrobeniusBall":
            pytest.skip("Only applies to FrobeniusBall")

        if shifted_nodes and shifted_nodes.issubset(roots):
            assert mech_cfg.get("eta", 0.0) == 0.0, (
                f"All shifted_nodes {shifted_nodes} are roots — "
                f"FrobeniusBall eta must be 0, got {mech_cfg['eta']}"
            )
