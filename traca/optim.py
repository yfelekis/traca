"""
Adversarial minimax transport map optimizer.

Alternating min-max optimization loop:
    outer: minimize over τ (project onto ConstructiveClass after each step)
    inner: maximize over adversary (ΔW, ξ) (project onto ambiguity sets after each step)

The logged scalar objective is:
    L(τ_t) = (1/|I|) Σ_ι F_ι(τ_t, ΔW*(t), ξ*(t))
where ΔW*(t), ξ*(t) are the SHARED adversary iterates after the inner loop at step t.

Row-vector convention: X = U @ A.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from traca.constructive import ConstructiveClass
from traca.ambiguity import (
    MechanismAmbiguitySet, EnvironmentAmbiguitySet,
    GelbrichBall, FrobeniusEmpirical, EntrywiseBox,
)
from traca.losses import GaussianLoss, EmpiricalLoss, numerical_gradient
from traca.stability import gating_matrix, perturbed_propagator
from traca.utils import as_rng, bures_sqrt, bundle_exo_means


# ---------------------------------------------------------------------------
# Convergence constants and helper
# ---------------------------------------------------------------------------

_REL_EPS = 1e-8    # denominator guard for zero-objective runs
_GAP_TOL = 1e-4    # gap-confirmed gate: convergence blocked while |gap| > this
_GAP_STOP = 1e-6   # gap-OR stopping arm: fires convergence when |gap| < this

_N_PGD_REFINE = 3  # PGD refinement steps after corner selection (EntrywiseBox)
_ENV_ETA_SCALE = 4  # μ step-size multiplier under corner enum
# Uniform-weight τ gradient over corners (equivalent to softmax T → ∞).
# Under corner enum, the GelbrichBall adversary uses sequential budget
# allocation: μ claims its share first (Euclidean projection at full eps),
# then CovProx (unchanged) gets the remaining W₂² budget.  CovProx is a
# proximal step that grabs the entire budget in ~2 steps, starving μ.
# Sequential allocation fixes this without touching CovProx: when μ is
# dominant (ATE, nonzero effective mean), μ claims the budget; when μ is
# dead (ATCE, zero mean), μ claims nothing and CovProx runs at full power.
# _ENV_ETA_SCALE speeds up μ convergence (eta_adv * 4 = 0.02 → ~28 steps).
#
# Uniform-weight τ gradient over corners: at symmetric saddles (e.g. ATE
# [-B, B]), opposing corners produce equal-and-opposite τ gradients.  Hard-max
# corner selection for the τ gradient causes oscillation: τ chases one corner,
# the other becomes worst, τ chases back → drift away from the optimal
# identity.  Uniform weighting (w_c = 1/k for all corners) fixes this: at a
# tie, equal weights → opposing gradients cancel → τ stays at identity.  When
# one corner dominates (asymmetric box or directional prior), its gradient is
# intrinsically larger and dominates the average → τ moves correctly.  The
# adversary still uses hard-max for its OWN corner selection (it needs the
# worst case); only the τ gradient computation changes.


def _enumerate_box_corners(box) -> list[np.ndarray]:
    """Enumerate all 2^k vertex matrices of an entrywise box.

    When box.delta is None (symmetric): corners are ±B[i,j] per nonzero entry.
    When box.delta is not None (shifted): corners are {delta-B, delta+B} per
    nonzero-B entry.

    k = number of structurally nonzero entries in B.  For paper configs
    k <= 2, so 2^k <= 4.  If k grows large, fall back to multi-start PGD
    (option b) instead.

    Returns list of (d, d) arrays, each a corner of the box.
    """
    B = box.B
    delta = box.delta
    nonzero = [(i, j) for i in range(B.shape[0]) for j in range(B.shape[1])
               if abs(B[i, j]) > 1e-15]
    k = len(nonzero)
    corners = []
    if delta is None:
        for signs in range(2 ** k):
            dW = np.zeros_like(B)
            for bit, (i, j) in enumerate(nonzero):
                sign = 1 if (signs >> bit) & 1 else -1
                dW[i, j] = sign * B[i, j]
            corners.append(dW)
    else:
        for signs in range(2 ** k):
            dW = np.zeros_like(B)
            for bit, (i, j) in enumerate(nonzero):
                if (signs >> bit) & 1:
                    dW[i, j] = delta[i, j] + B[i, j]  # high
                else:
                    dW[i, j] = delta[i, j] - B[i, j]  # low
            corners.append(dW)
    return corners


def _check_converged(
    previous_loss: float,
    current_loss: float,
    history_gap: list[float],
    tol: float,
    t: int,
    conv_window: int,
) -> tuple[bool, float]:
    """Single-shot convergence check with gap-confirmed gate.

    Convergence fires when ALL of the following hold:

      1. ``t >= conv_window``  (warm-up floor)
      2. ``|gap| < _GAP_TOL``  (gap-confirmed gate — saddle is near)
      3. EITHER  (a) ``rel_change < tol``   (loss stopped moving)
            OR   (b) ``|gap| < _GAP_STOP``  (saddle reached precisely)

    The gap-confirmed gate (condition 2) is a REQUIRED conjunct: convergence
    cannot fire while the gap exceeds 1e-4, regardless of rel_change.  This
    blocks premature stops on high-loss configs (e.g. lilucas_ew_full at
    ε=η=1.0: loss flat at 7.6 but gap still ~1.9e-3 and descending).

    Drift configs (gap=0 from iter 1) satisfy condition 2 trivially.
    Adversarial configs satisfy it once the gap descends below 1e-4.

    Returns ``(converged, rel_change)``.
    """
    rel_change = abs(previous_loss - current_loss) / (abs(current_loss) + _REL_EPS)
    gap_ok = len(history_gap) > 0 and abs(history_gap[-1]) < _GAP_TOL
    gap_tight = len(history_gap) > 0 and abs(history_gap[-1]) < _GAP_STOP
    fired = (rel_change < tol or gap_tight) and gap_ok and t >= conv_window
    return fired, rel_change


# ---------------------------------------------------------------------------
# CovProx helpers
# ---------------------------------------------------------------------------


def _to_observed(
    mu_exo: np.ndarray,
    Sigma_exo: np.ndarray,
    A_prime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert exogenous adversary params to observed space.

    μ_obs = μ_exo @ A'_ι
    Σ_obs = A'_ι.T @ Σ_exo @ A'_ι

    Row-vector convention: X = U @ A.
    """
    return mu_exo @ A_prime, A_prime.T @ Sigma_exo @ A_prime


def _interventional_exo_target(
    mu_t_exo: np.ndarray,
    Sigma_t_exo: np.ndarray,
    fixed: np.ndarray,
    J: list | tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply interventional frame to adversary exogenous params.

    Mirrors ``gaussian_joint()`` (lan_scm.py:314-322):
      - Mean:  zero entries at intervened nodes, then add _fixed.
      - Covariance:  zero rows AND columns at intervened nodes.

    This ensures the adversarial target describes the same intervened
    world as the source, so W₂²(τ_# P_s^ι, P_t^ι) = 0 when τ=I and
    the adversary is pinned to source.
    """
    mu = mu_t_exo.copy()
    Sigma = Sigma_t_exo.copy()
    if J:
        jj = list(J)
        mu[jj] = 0.0
        Sigma[jj, :] = 0.0
        Sigma[:, jj] = 0.0
    mu = mu + np.asarray(fixed, dtype=float)
    return mu, Sigma


def _cov_prox_update(
    Sigma_t_exo: np.ndarray,
    Sigma_pushed: np.ndarray,
    A_prime: np.ndarray,
    W: np.ndarray,
    dW: np.ndarray,
    R_iota: np.ndarray,
    eta: float,
) -> np.ndarray:
    """CovProx proximal gradient ascent step on the Bures surrogate.

    Operates on S = C_{ι,t}^{1/2} in endogenous space where
    C_{ι,t} = A'_ι.T @ Σ_t^exo @ A'_ι, then pulls back to exogenous space
    via F'_ι = (A'_ι)^{-1} = I - (W + ΔW) R_ι  (right-multiplication, LAN convention).

    The surrogate covariance term maximized over S is
        h(S) = ||S||_F^2 - 2 f_push ||S||_F
    where f_push = ||Σ_pushed^{1/2}||_F = sqrt(Tr(Σ_pushed)).

    Proximal gradient step:
        A_step  = (1 + 2η) S                          # gradient on ||S||_F^2
        S_new   = max(0, 1 - 2η f_push / ||A_step||_F) A_step  # prox for -2 f_push ||·||_F

    Parameters
    ----------
    Sigma_t_exo : (d, d) current adversarial covariance in exogenous space
    Sigma_pushed : (d, d) transported source covariance τ.T A_ι.T Σ_s A_ι τ
    A_prime : (d, d) perturbed propagator A'_ι(ΔW)
    W : (d, d) base structural matrix
    dW : (d, d) current mechanism perturbation
    R_iota : (d, d) gating matrix
    eta : float step size

    Returns
    -------
    (d, d) updated Σ_t in exogenous space (symmetric PSD)

    Notes
    -----
    Called once per intervention ι in fit_gaussian's shared-adversary update
    (CovProx Option A): each call uses the shared Sigma_t_exo and its own A'_ι
    and Sigma_pushed_ι; the per-ι outputs are averaged, then one Gelbrich
    projection is applied.  Retained as reference and tested by
    TestCovProxRoundTrip in test_optim.py.
    """
    d = W.shape[0]

    # Endogenous covariance under current A'_ι
    C = A_prime.T @ Sigma_t_exo @ A_prime      # (d, d)
    S = bures_sqrt(C)                           # C^{1/2}, symmetric PSD

    # f_push = ||Σ_pushed^{1/2}||_F = sqrt(Tr(Σ_pushed))
    f_push = float(np.sqrt(max(np.trace(Sigma_pushed), 0.0)))

    # Proximal gradient ascent
    A_step = (1.0 + 2.0 * eta) * S
    nA = float(np.linalg.norm(A_step, "fro"))
    coeff = max(0.0, 1.0 - 2.0 * eta * f_push / nA) if nA > 1e-12 else 0.0
    S_new = coeff * A_step                      # (d, d)
    C_new = S_new @ S_new.T                     # (d, d), PSD

    # Pull back to exogenous space: Σ_exo = F'^T C F'
    # F'_ι = (A'_ι)^{-1} = I - (W+ΔW) R_ι  (right-multiplication = LAN convention)
    F_prime = np.eye(d) - (W + dW) @ R_iota    # (A'_ι)^{-1}
    Sigma_new = F_prime.T @ C_new @ F_prime     # (d, d)

    # Symmetrize and PSD-clip
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma_new)
    return eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimConfig:
    """Configuration for the alternating min-max optimizer.

    Parameters
    ----------
    eta_tau : float
        Step size for the outer τ update (gradient descent).
    eta_adv : float
        Step size for the inner adversary update (gradient ascent).
    k_tau : int
        Number of τ (minimizer) steps per outer iteration.
        Matches DiRoCA's num_steps_min. Default 1.
    k_adv : int
        Number of inner adversary steps per outer iteration (constant).
        Matches DiRoCA's fixed num_steps_max — no ramp.
    n_iters : int
        Maximum outer iterations.
    tol : float
        Relative convergence tolerance on the outer objective.
        Convergence fires on the first iteration (after warm-up) where
        |L_prev - L| / (|L| + 1e-8) < tol.  Single-shot, matching
        DiRoCA opt_tools.py:1274-1284.
    conv_window : int
        Minimum number of outer iterations before convergence can fire
        (warm-up floor).  Matches DiRoCA's ``(epoch+1) >= 50`` guard.
    grad_mode : "numerical" | "analytic"
        Which gradient implementation to use. "numerical" is safer for
        debugging; "analytic" is faster for production.
    seed : int or None
        Random seed (unused currently; reserved for stochastic variants).
    """
    eta_tau: float = 1e-2
    eta_adv: float = 1e-2
    k_tau: int = 1
    k_adv: int = 5
    n_iters: int = 1000
    tol: float = 1e-6          # relative tolerance (DiRoCA-style symmetric)
    conv_window: int = 20
    grad_mode: Literal["numerical", "analytic"] = "analytic"
    grad_backend: Literal["analytic", "autograd"] = "autograd"
    covprox_aggregation: str = "norm_weighted"
    seed: int | None = None
    tau_init: str = "identity"
    tau_seed: int | None = None  # separate from data seed; drives init_tau("random")
    show_progress: bool = False  # live tqdm progress bar (off for tests/non-interactive)
    progress_desc: str = ""      # label for the progress bar (e.g. "ATE fold 2/5 ε=0.10 η=0.30")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """Result of an optimization run.

    Parameters
    ----------
    tau : (d, d) fitted transport map
    history_outer_loss : per-outer-iteration L(τ_t) values
    history_best_loss : running min_{t'<=t} L(τ_{t'})
    history_gap : per-outer-iteration primal-dual gap.
        L_adv (objective after adversary update, before τ update) minus
        L_tau (objective after τ update = history_outer_loss[t]).
        Should be ≥ 0 and shrink toward 0 at equilibrium.
    history_cond_tau : per-outer-iteration condition number of τ_t.
        For Markovian τ (diagonal): σ_max / σ_min of the diagonal entries.
        For semi-Markovian τ (block-diagonal): np.linalg.cond of the full matrix.
        A value near 1 means the map is close to a scaling; large values mean
        some shifted nodes are being scaled very differently from others, which
        can signal ill-conditioning of the transport problem.
    initial_loss : L(τ_init) before any updates
    final_loss : L(τ) at termination
    converged : True if convergence criterion was met
    converged_gap_confirmed : True if, at the point convergence fired,
        the duality gap was also below gap_tol (1e-4). Acts as a safety
        confirmation that the symmetric criterion reached the saddle
        (gap ≈ 0) rather than merely plateauing.
    n_iters : actual number of outer iterations run
    dW_iota : list of one element — shared adversarial ΔW* at termination, shape (d,d)
    mu_t_iota : Gaussian only — list of one element, shared adversarial μ_t* (exogenous space)
    Sigma_t_iota : Gaussian only — list of one element, shared adversarial Σ_t* (exogenous space)
    Theta_iota : Empirical only — list of one element, shared adversarial Θ*, shape (N,d)
    """
    tau: np.ndarray
    history_outer_loss: list[float]
    history_best_loss: list[float]
    history_gap: list[float]
    history_cond_tau: list[float]
    initial_loss: float
    final_loss: float
    converged: bool
    n_iters: int
    dW_iota: list[np.ndarray]
    mu_t_iota: list[np.ndarray] | None = None
    Sigma_t_iota: list[np.ndarray] | None = None
    Theta_iota: list[np.ndarray] | None = None
    converged_gap_confirmed: bool = False


# ---------------------------------------------------------------------------
# Gaussian optimizer
# ---------------------------------------------------------------------------

def fit_gaussian(
    bundle,
    ambiguity_W: MechanismAmbiguitySet,
    ambiguity_env: GelbrichBall,
    constructive: ConstructiveClass,
    config: OptimConfig | None = None,
    query_family: list | None = None,
) -> OptResult:
    """Adversarial minimax transport map optimization — Gaussian objective.

    Minimizes max_{ΔW,μ_t,Σ_t} (1/|I|) Σ_ι F̃_ι^ρ(τ, ΔW, μ_t, Σ_t)
    via alternating projected gradient descent/ascent with a SINGLE SHARED adversary
    (ΔW, μ_t, Σ_t) across all interventions.

    Adversary updates follow the CovProx scheme, Option A:
    - (μ_t, Σ_t) are maintained in **exogenous** space (single shared pair).
    - Each inner step accumulates per-ι gradients for ΔW and μ_t, then applies
      ONE projection onto the ambiguity sets.
    - Σ_t: CovProx Option A — call _cov_prox_update once per ι (each using its
      own A'_ι and Σ_pushed_ι), average the exogenous outputs, then ONE Gelbrich
      projection.
    - μ_t gradient is computed in observed space and pulled back: g_exo = A'_ι @ g_obs.
    - ΔW gradient uses analytic grad_dW (or numerical fallback) with (μ_t^exo, Σ_t^exo) fixed.
    - Observed-space params for loss/gradient evaluation: convert on the fly per ι.

    The Bures surrogate F̃_ι^ρ is used for the outer (τ) gradient.

    Parameters
    ----------
    bundle : SCMBundle
    ambiguity_W : mechanism ambiguity set
    ambiguity_env : Gelbrich environment ambiguity set
    constructive : ConstructiveClass
    config : OptimConfig (default values used if None)
    query_family : list of (iota_idx, O) pairs or None
        If provided, uses Q-restricted losses.  None = full-joint.

    Returns
    -------
    OptResult
    """
    if config is None:
        config = OptimConfig()

    # Lazy torch import — only needed when grad_backend="autograd".
    # The default analytic path never imports torch.
    if config.grad_backend == "autograd":
        try:
            import torch as _torch
            from traca.losses_torch import (
                surrogate_torch as _surrogate_torch,
                exact_loss_torch as _exact_loss_torch,
                perturbed_propagator_torch as _pp_torch,
            )
        except ImportError:
            raise ImportError(
                "grad_backend='autograd' requires PyTorch. "
                "Install with: pip install torch"
            )

    loss_fn = GaussianLoss()
    W = bundle.W
    d = bundle.d
    n_iota = bundle.n_interventions()
    mu_s_effs = bundle_exo_means(bundle)   # per-intervention effective exogenous means
    Sigma_s_raw = bundle.noise_cov

    # Per-intervention interventional data (for applying the same frame to
    # both source and target sides of the loss).
    _fixeds = [bundle.intervened_scms[i]._fixed for i in range(n_iota)]
    _Js     = [bundle.intervened_scms[i]._J     for i in range(n_iota)]

    # Per-intervention effective source covariance: zero rows/cols at
    # intervened nodes, mirroring gaussian_joint() (lan_scm.py:318-319).
    Sigma_s_effs = []
    for i in range(n_iota):
        S = Sigma_s_raw.copy()
        if _Js[i]:
            jj = list(_Js[i])
            S[jj, :] = 0.0
            S[:, jj] = 0.0
        Sigma_s_effs.append(S)

    # Build per-intervention propagators and gating matrices
    A_iotas, R_iotas = _build_propagators(bundle)

    # Determine which (iota, O) pairs to use
    pairs = _resolve_pairs(query_family, n_iota, d, bundle)

    # Initialize τ and shared adversary variables.
    # A single (ΔW, μ_t, Σ_t) is shared across all interventions, implementing
    # max_ξ (1/|I|) Σ_ι F_ι(τ, ξ) rather than (1/|I|) Σ_ι max_{ξ_ι} F_ι(τ, ξ_ι).
    #
    # Adversary (μ_t, Σ_t) is maintained in EXOGENOUS space per CovProx.
    # ΔW is initialized at the boundary of the mechanism ambiguity ball (random direction)
    # so gradient ascent starts from a non-trivial point with nonzero gradient.
    rng = as_rng(config.seed)
    tau = constructive.init_tau(mode=config.tau_init, rng=config.tau_seed)

    # EntrywiseBox corner enumeration: precompute all 2^k corners once.
    # For non-EntrywiseBox sets (FrobeniusBall, etc.), pure PGD is kept.
    _use_corners = isinstance(ambiguity_W, EntrywiseBox)
    _corners: list[np.ndarray] | None = None
    if _use_corners:
        _corners = _enumerate_box_corners(ambiguity_W)

    # k_adv is set per-config to satisfy two competing requirements:
    # - Symmetric saddles (ATE): low k_adv preserves corner symmetry so
    #   uniform-weight τ gradients cancel → τ stays at identity.
    # - Directional cases (ATCE): high k_adv lets the env adversary converge
    #   to the true worst case, enabling τ to beat identity.
    _k_adv = config.k_adv
    _eta_env = config.eta_adv * _ENV_ETA_SCALE if _use_corners else config.eta_adv

    adv = {
        "dW":      np.zeros((d, d)) if _use_corners
                   else ambiguity_W.project(rng.standard_normal((d, d))),
        "mu_t":    bundle.noise_mean.copy(),  # adversary in raw exogenous space
        "Sigma_t": Sigma_s_raw.copy(),
    }

    def compute_loss(tau_, adv_):
        losses = []
        for iota_idx, O in pairs:
            A_prime = perturbed_propagator(W, adv_["dW"], R_iotas[iota_idx])
            mu_t_int, Sigma_t_int = _interventional_exo_target(
                adv_["mu_t"], adv_["Sigma_t"],
                _fixeds[iota_idx], _Js[iota_idx],
            )
            mu_t_obs, Sigma_t_obs = _to_observed(mu_t_int, Sigma_t_int, A_prime)
            mu_s_i = mu_s_effs[iota_idx]
            Sigma_s_i = Sigma_s_effs[iota_idx]
            if O is None:
                l = loss_fn.value(
                    tau_, adv_["dW"], W, A_iotas[iota_idx], R_iotas[iota_idx],
                    mu_s_i, Sigma_s_i, mu_t_obs, Sigma_t_obs,
                )
            else:
                l = _restrict_loss_gaussian(
                    tau_, adv_["dW"], W, A_iotas[iota_idx], R_iotas[iota_idx],
                    mu_s_i, Sigma_s_i, mu_t_obs, Sigma_t_obs, O,
                )
            losses.append(l)
        return float(np.mean(losses))

    def _adv_update(tau_):
        """Aggregate gradient across all (ι, O) pairs, then ONE update + ONE projection."""
        n_pairs = len(pairs)
        Sigma_candidates = []
        f_push_weights: list[float] = []

        # --- EntrywiseBox corner selection for dW ---
        # Evaluate loss at all 2^k corners, select the worst.  This prevents
        # corner lock-in (see CLAUDE.md changelog).  The gradient step below
        # then refines from the selected corner using g_dW_total (no extra cost).
        if _use_corners:
            best_corner_loss = -np.inf
            for c in _corners:
                l = compute_loss(tau_, {**adv, "dW": c})
                if l > best_corner_loss:
                    best_corner_loss = l
                    adv["dW"] = c.copy()

        if config.grad_backend == "autograd":
            # ΔW and μ_t: one batched autograd backward through surrogate.
            # μ_t is in exogenous space; autograd handles the chain rule
            # μ_exo → interventional zeroing+_fixed → μ_obs = μ_int @ A'_ι → surrogate.
            W_t       = _torch.tensor(W,            dtype=_torch.float64)
            Sigma_t_exo_t = _torch.tensor(adv["Sigma_t"], dtype=_torch.float64)
            tau_t_fixed   = _torch.tensor(tau_,     dtype=_torch.float64)
            dW_t      = _torch.tensor(adv["dW"],    dtype=_torch.float64, requires_grad=True)
            mu_t_exo_t = _torch.tensor(adv["mu_t"], dtype=_torch.float64, requires_grad=True)

            loss_sum = _torch.zeros(1, dtype=_torch.float64)
            for iota_idx, O in pairs:
                Ri_t  = _torch.tensor(R_iotas[iota_idx], dtype=_torch.float64)
                Ai_t  = _torch.tensor(A_iotas[iota_idx], dtype=_torch.float64)
                mu_s_i_t = _torch.tensor(mu_s_effs[iota_idx], dtype=_torch.float64)
                Sigma_s_i_t = _torch.tensor(Sigma_s_effs[iota_idx], dtype=_torch.float64)
                A_prime_t  = _pp_torch(W_t, dW_t, Ri_t)
                # Apply interventional frame to target: zero intervened entries, add _fixed
                mu_t_int_t = mu_t_exo_t.clone()
                Sigma_t_int_t = Sigma_t_exo_t.clone()
                if _Js[iota_idx]:
                    jj = list(_Js[iota_idx])
                    # Zero intervened entries (preserves grad through non-zeroed entries)
                    mask_mu = _torch.ones(d, dtype=_torch.float64)
                    mask_mu[jj] = 0.0
                    mu_t_int_t = mu_t_int_t * mask_mu
                    mask_cov = _torch.ones((d, d), dtype=_torch.float64)
                    mask_cov[jj, :] = 0.0
                    mask_cov[:, jj] = 0.0
                    Sigma_t_int_t = Sigma_t_int_t * mask_cov
                fixed_t = _torch.tensor(_fixeds[iota_idx], dtype=_torch.float64)
                mu_t_int_t = mu_t_int_t + fixed_t
                mu_t_obs_t = mu_t_int_t @ A_prime_t
                Sigma_t_obs_t = A_prime_t.T @ Sigma_t_int_t @ A_prime_t
                loss_sum = loss_sum + _surrogate_torch(
                    tau_t_fixed, dW_t, W_t, Ai_t, Ri_t,
                    mu_s_i_t, Sigma_s_i_t, mu_t_obs_t, Sigma_t_obs_t,
                )
            (loss_sum / n_pairs).backward()
            g_dW_total     = np.nan_to_num(dW_t.grad.detach().numpy().copy(), nan=0.0)
            g_mu_exo_total = np.nan_to_num(mu_t_exo_t.grad.detach().numpy().copy(), nan=0.0)

            # Σ_t: CovProx is a proximal step, not a gradient — unaffected by grad_backend.
            for iota_idx, O in pairs:
                Ai     = A_iotas[iota_idx]
                Ri     = R_iotas[iota_idx]
                Sigma_s_i = Sigma_s_effs[iota_idx]
                A_prime = perturbed_propagator(W, adv["dW"], Ri)
                B = Ai @ tau_
                Sigma_pushed_i = B.T @ Sigma_s_i @ B
                f_push_weights.append(float(np.sqrt(max(np.trace(Sigma_pushed_i), 0.0))))
                Sigma_candidates.append(_cov_prox_update(
                    adv["Sigma_t"], Sigma_pushed_i, A_prime,
                    W, adv["dW"], Ri, config.eta_adv,
                ))

        else:
            # analytic / numerical path (default)
            g_dW_total     = np.zeros((d, d))
            g_mu_exo_total = np.zeros(d)

            for iota_idx, O in pairs:
                Ai = A_iotas[iota_idx]
                Ri = R_iotas[iota_idx]
                mu_s_i = mu_s_effs[iota_idx]
                Sigma_s_i = Sigma_s_effs[iota_idx]
                A_prime = perturbed_propagator(W, adv["dW"], Ri)
                mu_t_int, Sigma_t_int = _interventional_exo_target(
                    adv["mu_t"], adv["Sigma_t"],
                    _fixeds[iota_idx], _Js[iota_idx],
                )
                mu_t_obs, Sigma_t_obs = _to_observed(mu_t_int, Sigma_t_int, A_prime)

                # ΔW gradient: analytic (surrogate) or numerical
                if config.grad_mode == "analytic":
                    g_dW = loss_fn.grad_dW(
                        tau_, adv["dW"], W, Ai, Ri,
                        mu_s_i, Sigma_s_i, mu_t_obs, Sigma_t_obs,
                    )
                else:
                    def _f_dw(dw, _i=iota_idx):
                        _Ap = perturbed_propagator(W, dw, R_iotas[_i])
                        _mi, _Si = _interventional_exo_target(
                            adv["mu_t"], adv["Sigma_t"],
                            _fixeds[_i], _Js[_i],
                        )
                        _mo, _So = _to_observed(_mi, _Si, _Ap)
                        return loss_fn.value(tau_, dw, W, A_iotas[_i], R_iotas[_i],
                                             mu_s_effs[_i], Sigma_s_effs[_i], _mo, _So)
                    g_dW = numerical_gradient(_f_dw, adv["dW"])
                g_dW_total += g_dW

                # μ_t gradient: compute in observed space, pull back to exogenous
                # g_mu_exo = A'_ι @ g_mu_obs  (chain rule: μ_obs = μ_exo @ A'_ι)
                g_mu_obs = loss_fn.grad_mu_t(
                    tau_, adv["dW"], W, Ai, Ri, mu_s_i, Sigma_s_i,
                    mu_t_obs, Sigma_t_obs,
                )
                g_mu_exo_total += A_prime @ g_mu_obs

                # Σ_t: CovProx proximal step.
                B = Ai @ tau_
                Sigma_pushed_i = B.T @ Sigma_s_i @ B
                f_push_weights.append(float(np.sqrt(max(np.trace(Sigma_pushed_i), 0.0))))
                Sigma_candidates.append(_cov_prox_update(
                    adv["Sigma_t"], Sigma_pushed_i, A_prime,
                    W, adv["dW"], Ri, config.eta_adv,
                ))

            g_dW_total     /= n_pairs
            g_mu_exo_total /= n_pairs

        # CovProx aggregation across interventions (always runs, unchanged).
        if config.covprox_aggregation == "norm_weighted":
            Sigma_t_candidate = (2.0 / n_pairs) * sum(
                w * S for w, S in zip(f_push_weights, Sigma_candidates)
            )
        elif config.covprox_aggregation == "weighted_mean":
            total_w = sum(f_push_weights)
            if total_w > 1e-12:
                Sigma_t_candidate = sum(
                    w * S for w, S in zip(f_push_weights, Sigma_candidates)
                ) / total_w
            else:
                Sigma_t_candidate = np.mean(Sigma_candidates, axis=0)
        else:  # "mean"
            Sigma_t_candidate = np.mean(Sigma_candidates, axis=0)

        # ONE update + ONE projection per variable.
        # dW: corner-selected → PGD refinement; pure PGD otherwise.
        if _use_corners:
            # PGD refinement from corner-selected dW.
            for _ in range(_N_PGD_REFINE):
                g_refine = np.zeros((d, d))
                for iota_idx, O in pairs:
                    Ai = A_iotas[iota_idx]
                    Ri = R_iotas[iota_idx]
                    A_prime = perturbed_propagator(W, adv["dW"], Ri)
                    mu_t_int, Sigma_t_int = _interventional_exo_target(
                        adv["mu_t"], adv["Sigma_t"],
                        _fixeds[iota_idx], _Js[iota_idx],
                    )
                    mu_t_obs, Sigma_t_obs = _to_observed(
                        mu_t_int, Sigma_t_int, A_prime)
                    if config.grad_mode == "analytic":
                        g = loss_fn.grad_dW(
                            tau_, adv["dW"], W, Ai, Ri,
                            mu_s_effs[iota_idx], Sigma_s_effs[iota_idx],
                            mu_t_obs, Sigma_t_obs,
                        )
                    else:
                        def _f_ref(dw, _i=iota_idx):
                            _Ap = perturbed_propagator(W, dw, R_iotas[_i])
                            _mi, _Si = _interventional_exo_target(
                                adv["mu_t"], adv["Sigma_t"],
                                _fixeds[_i], _Js[_i],
                            )
                            _mo, _So = _to_observed(_mi, _Si, _Ap)
                            return loss_fn.value(
                                tau_, dw, W, A_iotas[_i], R_iotas[_i],
                                mu_s_effs[_i], Sigma_s_effs[_i], _mo, _So)
                        g = numerical_gradient(_f_ref, adv["dW"])
                    g_refine += g
                g_refine /= n_pairs
                adv["dW"] = ambiguity_W.project(
                    adv["dW"] + config.eta_adv * g_refine)

            # Sequential budget allocation: μ claims first, CovProx gets the rest.
            # W₂² = ‖μ−μ_s‖² + Bures²(Σ,Σ_s), so the two terms are additive.
            # Step 1: μ gradient step → project μ onto the full eps-ball.
            mu_t_new = adv["mu_t"] + _eta_env * g_mu_exo_total
            # Invariance pinning on μ (non-shifted coords stay at source).
            _sr = set(ambiguity_env.shifted_rows)
            for i in range(d):
                if i not in _sr:
                    mu_t_new[i] = ambiguity_env.mu_s[i]
            # Euclidean projection of shifted coords onto ‖μ−μ_s‖ ≤ eps.
            mu_diff = mu_t_new - ambiguity_env.mu_s
            mu_dist = float(np.linalg.norm(mu_diff))
            if mu_dist > ambiguity_env.eps:
                mu_proj = ambiguity_env.mu_s + mu_diff * (ambiguity_env.eps / mu_dist)
            else:
                mu_proj = mu_t_new
            # Step 2: remaining budget for Σ.
            mu_budget = float(np.linalg.norm(mu_proj - ambiguity_env.mu_s) ** 2)
            remaining_eps = np.sqrt(max(ambiguity_env.eps ** 2 - mu_budget, 0.0))
            # Step 3: project CovProx's Sigma_candidate onto remaining budget.
            # Reuse GelbrichBall.project with mean fixed at mu_proj: the mean
            # term vanishes, so bisection constrains only Bures²(Σ, Σ_s).
            _sigma_ball = GelbrichBall(
                mu_s=mu_proj, Sigma_s=ambiguity_env.Sigma_s,
                eps=remaining_eps, shifted_rows=ambiguity_env.shifted_rows,
            )
            _, Sigma_proj = _sigma_ball.project(mu_proj, Sigma_t_candidate)
            adv["mu_t"] = mu_proj
            adv["Sigma_t"] = Sigma_proj
        else:
            adv["dW"] = ambiguity_W.project(
                adv["dW"] + config.eta_adv * g_dW_total)
            mu_t_new = adv["mu_t"] + _eta_env * g_mu_exo_total
            adv["mu_t"], adv["Sigma_t"] = ambiguity_env.project(
                mu_t_new, Sigma_t_candidate)

    # Warm up adversary before the main loop so it starts from a non-trivial state.
    for _ in range(_k_adv):
        _adv_update(tau)

    # DiRoCA-style constant-schedule loop: fixed k_adv inner steps per outer
    # iteration, return the last τ. No ramp, no deferred baseline, no global-best
    # tracker. Convergence: symmetric relative-change test on the outer objective
    # (matches DiRoCA opt_tools.py:1274-1284).
    initial_loss = compute_loss(tau, adv)
    previous_loss = initial_loss
    history_outer = []
    history_best = []
    history_gap = []
    history_cond = []
    running_min = initial_loss  # tracked for history_best only, not convergence

    pbar = None
    if config.show_progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=config.n_iters, desc=config.progress_desc or "fit_gaussian",
                         unit="it", leave=False)
        except ImportError:
            pass  # tqdm not installed — run silently

    for t in range(config.n_iters):
        # --- Inner adversary step (maximize over ΔW, μ_t, Σ_t) ---
        for _ in range(_k_adv):
            _adv_update(tau)

        # Snapshot: adversary's best response against current τ (before τ update).
        L_adv = compute_loss(tau, adv)

        # --- Outer τ step (minimize, k_tau steps — matches DiRoCA num_steps_min) ---
        for _ in range(config.k_tau):
            # When using corner enum, compute uniform-weighted τ gradient
            # across all corners.  This prevents saddle oscillation at symmetric
            # boxes: tied corners produce opposing gradients that cancel under
            # equal weights, so τ stays at identity.  When one corner's gradient
            # is intrinsically larger (asymmetric), it dominates the average.
            # For non-corner-enum (FrobeniusBall etc.), single-corner gradient.
            if _use_corners:
                # Uniform weights: 1/k for each of k corners
                weights = np.ones(len(_corners)) / len(_corners)
                dw_list = _corners
            else:
                weights = np.array([1.0])
                dw_list = [adv["dW"]]

            grad_tau_total = np.zeros_like(tau)
            for w_c, dw_c in zip(weights, dw_list):
                if w_c < 1e-12:
                    continue  # skip negligible corners
                if config.grad_backend == "autograd":
                    W_t       = _torch.tensor(W,       dtype=_torch.float64)
                    tau_t = _torch.tensor(tau, dtype=_torch.float64, requires_grad=True)
                    loss_sum = _torch.zeros(1, dtype=_torch.float64)
                    for iota_idx, O in pairs:
                        Ai = A_iotas[iota_idx]
                        Ri = R_iotas[iota_idx]
                        mu_s_i_t = _torch.tensor(mu_s_effs[iota_idx], dtype=_torch.float64)
                        Sigma_s_i_t = _torch.tensor(Sigma_s_effs[iota_idx], dtype=_torch.float64)
                        A_prime = perturbed_propagator(W, dw_c, Ri)
                        mu_t_int, Sigma_t_int = _interventional_exo_target(
                            adv["mu_t"], adv["Sigma_t"],
                            _fixeds[iota_idx], _Js[iota_idx],
                        )
                        mu_t_obs, Sigma_t_obs = _to_observed(mu_t_int, Sigma_t_int, A_prime)
                        Ai_t = _torch.tensor(Ai, dtype=_torch.float64)
                        Ri_t = _torch.tensor(Ri, dtype=_torch.float64)
                        dW_t = _torch.tensor(dw_c, dtype=_torch.float64)
                        mu_t_t    = _torch.tensor(mu_t_obs,    dtype=_torch.float64)
                        Sigma_t_t = _torch.tensor(Sigma_t_obs, dtype=_torch.float64)
                        loss_sum = loss_sum + _exact_loss_torch(
                            tau_t, dW_t, W_t, Ai_t, Ri_t,
                            mu_s_i_t, Sigma_s_i_t, mu_t_t, Sigma_t_t,
                        )
                    (loss_sum / len(pairs)).backward()
                    g = tau_t.grad.detach().numpy()
                    g = np.nan_to_num(g, nan=0.0)
                    grad_tau_total += w_c * g
                else:
                    g = np.zeros_like(tau)
                    for iota_idx, O in pairs:
                        Ai = A_iotas[iota_idx]
                        Ri = R_iotas[iota_idx]
                        mu_s_i = mu_s_effs[iota_idx]
                        Sigma_s_i = Sigma_s_effs[iota_idx]
                        A_prime = perturbed_propagator(W, dw_c, Ri)
                        mu_t_int, Sigma_t_int = _interventional_exo_target(
                            adv["mu_t"], adv["Sigma_t"],
                            _fixeds[iota_idx], _Js[iota_idx],
                        )
                        mu_t_obs, Sigma_t_obs = _to_observed(mu_t_int, Sigma_t_int, A_prime)
                        if config.grad_mode == "analytic":
                            gi = loss_fn.grad_tau(
                                tau, dw_c, W, Ai, Ri, mu_s_i, Sigma_s_i,
                                mu_t_obs, Sigma_t_obs,
                            )
                        else:
                            def f_tau(t_, _dw=dw_c, _Ai=Ai, _Ri=Ri,
                                      _mo=mu_t_obs, _So=Sigma_t_obs,
                                      _mu_s_i=mu_s_i, _Sigma_s_i=Sigma_s_i):
                                return loss_fn.surrogate(t_, _dw, W, _Ai, _Ri,
                                                     _mu_s_i, _Sigma_s_i, _mo, _So)
                            gi = numerical_gradient(f_tau, tau)
                        g += gi
                    g /= len(pairs)
                    grad_tau_total += w_c * g

            tau = constructive.project(tau - config.eta_tau * grad_tau_total)

        # --- Log objective and condition number ---
        L = compute_loss(tau, adv)
        history_outer.append(L)
        history_gap.append(L_adv - L)

        if L < running_min:
            running_min = L
        history_best.append(running_min)
        history_cond.append(float(np.linalg.cond(tau)))

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str("loss=%.2e gap=%.2e" % (L, L_adv - L))

        # --- Convergence: single-shot with gap-OR ---
        # Fires when (rel_change < tol OR |gap| < _GAP_STOP) AND t >= conv_window.
        # The gap-OR arm catches drift-dominated configs (|L| ≈ _REL_EPS) where
        # rel_change stays inflated despite the saddle being reached.
        converged, _ = _check_converged(
            previous_loss, L, history_gap, config.tol, t, config.conv_window,
        )
        previous_loss = L
        if converged:
            break
    else:
        converged = False

    if pbar is not None:
        pbar.close()

    final_loss = float(history_outer[-1]) if history_outer else initial_loss
    gap_at_exit = float(history_gap[-1]) if history_gap else 0.0
    gap_confirmed = abs(gap_at_exit) < _GAP_TOL

    return OptResult(
        tau=tau,
        history_outer_loss=history_outer,
        history_best_loss=history_best,
        history_gap=history_gap,
        history_cond_tau=history_cond,
        initial_loss=initial_loss,
        final_loss=final_loss,
        converged=converged,
        converged_gap_confirmed=gap_confirmed,
        n_iters=len(history_outer),
        dW_iota=[adv["dW"]],
        mu_t_iota=[adv["mu_t"]],
        Sigma_t_iota=[adv["Sigma_t"]],
        Theta_iota=None,
    )


# ---------------------------------------------------------------------------
# Empirical optimizer
# ---------------------------------------------------------------------------

def fit_empirical(
    bundle,
    ambiguity_W: MechanismAmbiguitySet,
    ambiguity_env: FrobeniusEmpirical,
    constructive: ConstructiveClass,
    config: OptimConfig | None = None,
    query_family: list | None = None,
) -> OptResult:
    """Adversarial minimax transport map optimization — Empirical objective.

    Minimizes max_{ΔW,Θ} (1/|I|) Σ_ι F_ι^U(τ, ΔW, Θ)
    via alternating projected gradient descent/ascent with a SINGLE SHARED
    adversary (ΔW, Θ) across all interventions.

    Parameters
    ----------
    bundle : SCMBundle
    ambiguity_W : mechanism ambiguity set
    ambiguity_env : FrobeniusEmpirical environment ambiguity set
    constructive : ConstructiveClass
    config : OptimConfig (default values used if None)
    query_family : list of (iota_idx, O) pairs or None

    Returns
    -------
    OptResult
    """
    if config is None:
        config = OptimConfig()

    loss_fn = EmpiricalLoss()
    W = bundle.W
    d = bundle.d
    n_iota = bundle.n_interventions()

    A_iotas, R_iotas = _build_propagators(bundle)
    pairs = _resolve_pairs(query_family, n_iota, d, bundle)

    # DiRoCA shared-residual protocol: reuse the observational noise samples
    # U^(0) as the base residuals for ALL interventions during training.
    # The intervention effect enters through A_ι, R_ι, AND the effective
    # exogenous mean (zeroed intervened columns + fixed values).
    # The adversary Θ is a single shared perturbation — it must act on the
    # same base residuals across all interventions.
    obs_idx = bundle.interventions.index({})
    ns = bundle.noise_samples
    U_obs = ns[obs_idx] if isinstance(ns, dict) else ns[obs_idx]

    # Per-intervention effective noise: zero intervened columns, add fixed row offset.
    # This matches gaussian_joint()'s mean logic applied to samples.
    U_effs: dict[int, np.ndarray] = {}
    for i in range(n_iota):
        scm_i = bundle.intervened_scms[i]
        U_eff = U_obs.copy()
        if scm_i._J:
            U_eff[:, list(scm_i._J)] = 0.0
        U_eff += scm_i._fixed[np.newaxis, :]
        U_effs[i] = U_eff

    # Initialize shared adversary at boundary of ambiguity balls to break
    # zero-gradient symmetry.  A single (ΔW, Θ) is shared across all interventions,
    # implementing max_ξ (1/|I|) Σ_ι F_ι rather than (1/|I|) Σ_ι max_{ξ_ι} F_ι.
    # N = bundle.n is uniform across all interventions (guaranteed by scm.bundle()).
    rng = as_rng(config.seed)
    tau = constructive.init_tau(mode=config.tau_init, rng=config.tau_seed)
    N = bundle.n

    # EntrywiseBox corner enumeration (same logic as fit_gaussian).
    _use_corners = isinstance(ambiguity_W, EntrywiseBox)
    _corners: list[np.ndarray] | None = None
    if _use_corners:
        _corners = _enumerate_box_corners(ambiguity_W)

    adv = {
        "dW":    np.zeros((d, d)) if _use_corners
                 else ambiguity_W.project(rng.standard_normal((d, d))),
        "Theta": ambiguity_env.project(rng.standard_normal((N, d))),
    }

    def compute_loss(tau_, adv_):
        losses = []
        for iota_idx, O in pairs:
            U_i = U_effs[iota_idx]
            if O is None:
                l = loss_fn.value(
                    tau_, adv_["dW"], adv_["Theta"],
                    W, A_iotas[iota_idx], R_iotas[iota_idx], U_i,
                )
            else:
                l = _restrict_loss_empirical(
                    tau_, adv_["dW"], adv_["Theta"],
                    W, A_iotas[iota_idx], R_iotas[iota_idx], U_i, O,
                )
            losses.append(l)
        return float(np.mean(losses))

    def _adv_update_empirical(tau_):
        """Aggregate gradient across all (ι, O) pairs, then ONE update + ONE projection."""
        # --- EntrywiseBox corner selection for dW ---
        if _use_corners:
            best_corner_loss = -np.inf
            for c in _corners:
                l = compute_loss(tau_, {"dW": c, "Theta": adv["Theta"]})
                if l > best_corner_loss:
                    best_corner_loss = l
                    adv["dW"] = c.copy()

        g_dW_total    = np.zeros((d, d))
        g_Theta_total = np.zeros((N, d))
        for iota_idx, O in pairs:
            Ai  = A_iotas[iota_idx]
            Ri  = R_iotas[iota_idx]
            U_i = U_effs[iota_idx]
            if config.grad_mode == "analytic":
                g_dW   = loss_fn.grad_dW(tau_, adv["dW"], adv["Theta"], W, Ai, Ri, U_i)
                g_Theta = loss_fn.grad_Theta(tau_, adv["dW"], adv["Theta"], W, Ai, Ri, U_i)
            else:
                def _f_dw(dw, _i=iota_idx):
                    return loss_fn.value(tau_, dw, adv["Theta"],
                                         W, A_iotas[_i], R_iotas[_i], U_effs[_i])
                def _f_th(th, _i=iota_idx):
                    return loss_fn.value(tau_, adv["dW"], th,
                                         W, A_iotas[_i], R_iotas[_i], U_effs[_i])
                g_dW   = numerical_gradient(_f_dw, adv["dW"])
                g_Theta = numerical_gradient(_f_th, adv["Theta"])
            g_dW_total    += g_dW
            g_Theta_total += g_Theta
        n_pairs = len(pairs)
        g_dW_total    /= n_pairs
        g_Theta_total /= n_pairs
        # dW: PGD refinement after corner selection, or pure PGD.
        if _use_corners:
            for _ in range(_N_PGD_REFINE):
                g_refine = np.zeros((d, d))
                for iota_idx, O in pairs:
                    Ai  = A_iotas[iota_idx]
                    Ri  = R_iotas[iota_idx]
                    U_i = U_effs[iota_idx]
                    if config.grad_mode == "analytic":
                        g = loss_fn.grad_dW(tau_, adv["dW"], adv["Theta"],
                                            W, Ai, Ri, U_i)
                    else:
                        def _f_ref(dw, _i=iota_idx):
                            return loss_fn.value(tau_, dw, adv["Theta"],
                                                 W, A_iotas[_i], R_iotas[_i],
                                                 U_effs[_i])
                        g = numerical_gradient(_f_ref, adv["dW"])
                    g_refine += g
                g_refine /= n_pairs
                adv["dW"] = ambiguity_W.project(
                    adv["dW"] + config.eta_adv * g_refine)
        else:
            adv["dW"] = ambiguity_W.project(
                adv["dW"] + config.eta_adv * g_dW_total)
        # Θ step: scale by N so the adversary traverses the FrobeniusEmpirical
        # ball (radius = ε√N) in O(ε/η_adv) steps, independent of N.
        # ∇_Θ L = O(1/√N) and the ball radius = O(√N), so raw η_adv gives
        # displacement O(η/√N) per step — astronomically slow for large N.
        # Multiplying by N yields displacement O(η√N) per step, matching the
        # ball scale.  Derived from: K·η_eff·‖∇Θ‖ = ε√N  ⟹  η_eff = η·N.
        adv["Theta"] = ambiguity_env.project(adv["Theta"] + config.eta_adv * N * g_Theta_total)

    # Warm up adversary before the main loop.
    for _ in range(config.k_adv):
        _adv_update_empirical(tau)

    # DiRoCA-style constant-schedule loop: same structure as fit_gaussian.
    initial_loss = compute_loss(tau, adv)
    previous_loss = initial_loss
    history_outer = []
    history_best = []
    history_gap = []
    history_cond = []
    running_min = initial_loss  # tracked for history_best only

    pbar = None
    if config.show_progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=config.n_iters, desc=config.progress_desc or "fit_empirical",
                         unit="it", leave=False)
        except ImportError:
            pass

    for t in range(config.n_iters):
        # --- Inner adversary step ---
        for _ in range(config.k_adv):
            _adv_update_empirical(tau)

        # Snapshot: adversary's best response against current τ (before τ update).
        L_adv = compute_loss(tau, adv)

        # --- Outer τ step (k_tau steps — matches DiRoCA num_steps_min) ---
        # Uniform-weight τ gradient over corners (same logic as fit_gaussian).
        for _ in range(config.k_tau):
            if _use_corners:
                weights = np.ones(len(_corners)) / len(_corners)
                dw_list = _corners
            else:
                weights = np.array([1.0])
                dw_list = [adv["dW"]]

            grad_tau_total = np.zeros_like(tau)
            for w_c, dw_c in zip(weights, dw_list):
                if w_c < 1e-12:
                    continue
                g = np.zeros_like(tau)
                for iota_idx, O in pairs:
                    Ai = A_iotas[iota_idx]
                    Ri = R_iotas[iota_idx]
                    U_i = U_effs[iota_idx]
                    if config.grad_mode == "analytic":
                        gi = loss_fn.grad_tau(
                            tau, dw_c, adv["Theta"],
                            W, Ai, Ri, U_i,
                        )
                    else:
                        def f_tau(t_, _dw=dw_c, _U_i=U_i):
                            return loss_fn.value(t_, _dw, adv["Theta"], W, Ai, Ri, _U_i)
                        gi = numerical_gradient(f_tau, tau)
                    g += gi
                g /= len(pairs)
                grad_tau_total += w_c * g

            tau = constructive.project(tau - config.eta_tau * grad_tau_total)

        # --- Log ---
        L = compute_loss(tau, adv)
        history_outer.append(L)
        history_gap.append(L_adv - L)

        if L < running_min:
            running_min = L
        history_best.append(running_min)
        history_cond.append(float(np.linalg.cond(tau)))

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str("loss=%.2e gap=%.2e" % (L, L_adv - L))

        # --- Convergence: single-shot with gap-OR ---
        # Same logic as fit_gaussian — see _check_converged.
        converged, _ = _check_converged(
            previous_loss, L, history_gap, config.tol, t, config.conv_window,
        )
        previous_loss = L
        if converged:
            break
    else:
        converged = False

    if pbar is not None:
        pbar.close()

    final_loss = float(history_outer[-1]) if history_outer else initial_loss
    gap_at_exit = float(history_gap[-1]) if history_gap else 0.0
    gap_confirmed = abs(gap_at_exit) < _GAP_TOL

    return OptResult(
        tau=tau,
        history_outer_loss=history_outer,
        history_best_loss=history_best,
        history_gap=history_gap,
        history_cond_tau=history_cond,
        initial_loss=initial_loss,
        final_loss=final_loss,
        converged=converged,
        converged_gap_confirmed=gap_confirmed,
        n_iters=len(history_outer),
        dW_iota=[adv["dW"]],
        mu_t_iota=None,
        Sigma_t_iota=None,
        Theta_iota=[adv["Theta"]],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_propagators(bundle):
    """Build per-intervention A_ι and R_ι from bundle."""
    d = bundle.d
    var_names = bundle.scm.var_names
    A_iotas = []
    R_iotas = []
    for i, iv in enumerate(bundle.interventions):
        scm_do = bundle.intervened_scms[i]
        A_iotas.append(scm_do.A)
        if iv:
            intervened_nodes = [
                var_names.index(k) if isinstance(k, str) else int(k)
                for k in iv.keys()
            ]
        else:
            intervened_nodes = []
        R_iotas.append(gating_matrix(d, intervened_nodes))
    return A_iotas, R_iotas


def _resolve_pairs(query_family, n_iota, d, bundle):
    """Resolve query_family to list of (iota_idx, O) pairs.

    O=None means full post-interventional (all variables not in J_ι).
    """
    if query_family is None:
        return [(i, None) for i in range(n_iota)]
    return [(int(iota_idx), tuple(O) if O is not None else None)
            for iota_idx, O in query_family]


def _restrict_loss_gaussian(tau, dW, W, A_iota, R_iota, mu_s, Sigma_s,
                             mu_t, Sigma_t, O) -> float:
    """Q-restricted Gaussian loss: project onto output coordinates O."""
    # Delegate to query module if available; else compute inline
    try:
        from traca.query import F_iota_O_rho
        return F_iota_O_rho(tau, dW, W, A_iota, R_iota, mu_s, Sigma_s,
                            mu_t, Sigma_t, O)
    except ImportError:
        return GaussianLoss().value(tau, dW, W, A_iota, R_iota, mu_s, Sigma_s,
                                    mu_t, Sigma_t)


def _restrict_loss_empirical(tau, dW, Theta, W, A_iota, R_iota, U_s, O) -> float:
    """Q-restricted empirical loss: project onto output coordinates O."""
    try:
        from traca.query import F_iota_O_U
        return F_iota_O_U(tau, dW, Theta, W, A_iota, R_iota, U_s, O)
    except ImportError:
        return EmpiricalLoss().value(tau, dW, Theta, W, A_iota, R_iota, U_s)
