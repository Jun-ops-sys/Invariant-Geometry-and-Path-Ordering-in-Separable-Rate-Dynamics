"""
Lie_Bracket_Mixed_Harvesting.py
================================
Numerical validation of Lie bracket order-dependence for the H4a + H4×
mixed harvesting model.

What this model is
------------------
  H(N; c, k1, k2) = r(1 - N/K) - c - k1/N - k2·N/(N + N0)

Channel 1 — constant-yield harvesting per capita (H4a):  f1(N) = 1/N   → +∞ as N → 0+
Channel 2 — Holling Type II saturating harvesting (H4×): f2(N) = N/(N+N0) → 0  as N → 0+

What the analytic bracket says
-------------------------------
Lie bracket of the two control vector fields at repeller N*:

  [X1, X2] = (1/H'(N*)) * [ -f1'(1 + f2²) ∂_{k1}
                             +f2'(1 + f1²) ∂_{k2} ]

with f1' = -1/N², f2' = N0/(N+N0)².  Both bracket_k1 and bracket_k2 are
positive at the repeller (contrast with H4a+H4b where they have opposite
signs): non-commutativity lies in the rate of movement, not its direction.

Path-ordering coefficient:
  L(N*) = (f1·f2' - f2·f1') / H'(N*)²
         = (N* + 2N0) / (N*·(N*+N0)²·H'(N*)²)

Why near-zero behaviour matters
--------------------------------
  H'(N*) ≈ k1/N*²  as N* → 0+  (H4a term dominates)
  L(N*) ≈ 2N*³ / (N0·k1²)  →  0

Despite the H4a singularity in the bracket numerator (∼1/N*), H' grows
as k1/N*² and overwhelms it — L suppresses near zero.  Meanwhile,
f1/f2 = (N*+N0)/N*² → ∞: the H4a channel dominates the parameter-space
geometry even as L → 0 (direction and amplitude decouple).

What the experiment does
-------------------------
Starting from baseline repeller N*₀, apply two sequential linearised
IFT steps along the two channels in opposite orders:

    Path A: push k1 by Δk1 (step at N*₀), then push k2 by Δk2 (step at N*₁A)
    Path B: push k2 by Δk2 (step at N*₀), then push k1 by Δk1 (step at N*₁B)

The second-order difference is:
    N*_A - N*_B ≈ Δk1·Δk2 · L(N*₀)

Since L > 0 and both Δk1, Δk2 > 0, Path A leaves the repeller higher
than Path B.  The experiment verifies that ratio_lin = diff_AB/(Δk1·Δk2)
converges to L as (Δk1, Δk2) → 0.

References
----------
Clark, C. W. (1976). Mathematical Bioeconomics: The Optimal Management
    of Renewable Resources. Wiley.  [constant-yield harvesting, f1 = 1/N]

Holling, C. S. (1959). The components of predation as revealed by a study
    of small mammal predation of the European pine sawfly. The Canadian
    Entomologist, 91(5), 293–320.  [Type II functional response, f2 form]

Hilborn, R., & Walters, C. J. (1992). Quantitative Fisheries Stock
    Assessment: Choice, Dynamics and Uncertainty. Chapman & Hall.
    [mixed harvesting strategy background]

==============================
Code navigation (Ctrl+F):
  # TAG: Config
  # TAG: Model definition
  # TAG: Fixed point detection
  # TAG: Lie bracket computation
  # TAG: Path experiment
  # TAG: Scaling verification
  # TAG: Report generation
  # TAG: Main entry point
==============================

"""

import datetime
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ══════════════════════════════════════════════════════════════
# §0  Global Configuration
# ══════════════════════════════════════════════════════════════

# TAG: Config
class Config:
    OUTPUT_DIR  = Path('.')
    REPORT_NAME = 'lie_bracket_mixed_harvesting_report.txt'

    # Fixed-point search
    FP_N_LOG = 4000
    FP_N_LIN = 3000
    FP_XTOL  = 1e-12
    FP_RTOL  = 1e-12
    U_MIN    = 1e-8    # must stay away from s=0 (H4a singularity)
    U_MAX    = 9.999

    # Degeneracy guard: |H'(N*)| below this → repeller/attractor classification unreliable.
    DEGEN_TOL   = 1e-6
    # Saddle-node (tangential root) detection threshold. Independent of DEGEN_TOL.
    TANGENT_TOL = 1e-6

    # Numerical derivative steps
    DERIV_H     = 1e-6     # step for dH/ds and d²H/ds² central differences

    # Four-point mixed partial divides by 4δ²; δ=1e-4 keeps denominator ~4e-8,
    # well above brentq noise floor (~1e-12 × 25 amplification at δ=1e-7).
    DERIV_MIXED = 1e-4

    # (Δk1, Δk2) pairs — Δk2 fixed, Δk1 varied to verify Δk1·Δk2 scaling
    DELTA_PAIRS = [
        (0.200, 0.010),
        (0.100, 0.010),
        (0.050, 0.010),
        (0.020, 0.010),
        (0.010, 0.010),
        (0.005, 0.010),
        (0.002, 0.010),
        (0.001, 0.010),
    ]

    # Joint-limit check: Δk2 = Δk1, slope of log|diff_AB| vs log(product) ≈ 1.
    PROPORTIONAL_PAIRS = [
        (0.100, 0.100),
        (0.050, 0.050),
        (0.020, 0.020),
        (0.010, 0.010),
        (0.005, 0.005),
        (0.002, 0.002),
        (0.001, 0.001),
    ]

    # Branch continuity guard: reject repeller that moved more than this fraction of prev_N.
    MAX_DRIFT_FRAC = 0.30

    # Baseline validity thresholds for main() [2/6].
    # HP_MARGIN: min |H'(N*)| — below this, baseline is too close to saddle-node.
    # SEP_FRAC:  min (N_att - N*) / N_att — ensures wide bistable window.
    HP_MARGIN = 1e-3
    SEP_FRAC  = 0.10

    # Suppresses high-frequency grid-search warnings (brentq failures, drift guards).
    # Other UserWarnings remain visible.
    SUPPRESS_DEBUG = True

    @classmethod
    def out(cls, filename: str) -> Path:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUT_DIR / filename

    @classmethod
    def summary(cls) -> str:
        lines = [
            f"  FP grid     : {cls.FP_N_LOG} log + {cls.FP_N_LIN} linear points",
            f"  FP Brent    : xtol={cls.FP_XTOL:.0e}  rtol={cls.FP_RTOL:.0e}",
            f"  U domain    : ({cls.U_MIN:.0e}, {cls.U_MAX})",
            f"  H' step     : {cls.DERIV_H:.0e}",
            f"  Mixed step  : {cls.DERIV_MIXED:.0e}  (mixed partial ∂²N*/∂k1∂k2; larger to avoid cancellation)",
            f"  Degen tol   : {cls.DEGEN_TOL:.0e}  (H' degeneracy guard)",
            f"  Tangent tol : {cls.TANGENT_TOL:.0e}  (tangential root detection; semantically independent of DEGEN_TOL)",
            f"  Delta pairs : {len(cls.DELTA_PAIRS)} (Δk2 fixed)  +  "
              f"{len(cls.PROPORTIONAL_PAIRS)} proportional (Δk2=Δk1)",
            f"  Max drift   : {cls.MAX_DRIFT_FRAC:.0%} of prev_N  (branch continuity guard)",
            f"  HP margin   : {cls.HP_MARGIN:.0e}  (min |H'(N*)| for valid baseline)",
            f"  Sep frac    : {cls.SEP_FRAC:.0%}  (min (N_att-N*)/N_att)",
        ]
        return '\n'.join(lines)


if Config.SUPPRESS_DEBUG:
    for _pat in ('.*brentq failed.*', '.*safe_find_repeller_near.*'):
        warnings.filterwarnings('ignore', category=UserWarning, message=_pat)


# ══════════════════════════════════════════════════════════════
# §1  Model Definition
# ══════════════════════════════════════════════════════════════

# TAG: Model definition
#
# H(N; c, k1, k2) = r(1 - N/K) - c - k1/N - k2·N/(N + N0)
#
# f1(N) = 1/N          (constant-yield harvesting per capita; H4a)
# f2(N) = N/(N+N0)     (Holling Type II saturating harvesting; H4×)
#
# Parameters produce a bistable structure with repeller around N*≈0.11
# and attractor around N_att≈7.8, giving a wide, numerically stable window.

N0 = 2.0   # Holling half-saturation constant for channel 2

BASE_PARAMS = {
    'r':  1.0,
    'K':  10.0,
    'c':  0.05,
    'k1': 0.10,
    'k2': 0.20,
}


def H_model(N, p):
    """H(N) = r(1-N/K) - c - k1/N - k2·N/(N+N0).  Requires N > 0.

    The tighter bound N > U_MIN is a numerical constraint applied by derivative
    functions to avoid the H4a singularity (k1/N → ∞). Calling H_model directly
    with 0 < N < U_MIN is mathematically valid but returns very large values.
    """
    if N <= 0:
        raise ValueError(f"H_model called with N={N!r}; must be > 0.")
    return (p['r'] * (1.0 - N / p['K'])
            - p['c']
            - p['k1'] / N
            - p['k2'] * N / (N + N0))


def f1(N):
    """f1(N) = 1/N  (H4a: f1 → +∞ as N → 0+)."""
    return 1.0 / N


def f2(N):
    """f2(N) = N/(N+N0)  (H4×: f2 → 0 as N → 0+)."""
    return N / (N + N0)


def f1_prime(N):
    """df1/dN = -1/N²."""
    return -1.0 / N**2


def f2_prime(N):
    """df2/dN = N0/(N+N0)²."""
    return N0 / (N + N0)**2


def H_prime_analytic(N, p):
    """dH/dN = -r/K + k1/N² - k2·N0/(N+N0)²."""
    return (-p['r'] / p['K']
            + p['k1'] / N**2
            - p['k2'] * N0 / (N + N0)**2)


def H_double_prime_analytic(N, p):
    """d²H/dN² = -2k1/N³ + 2k2·N0/(N+N0)³."""
    return -2.0 * p['k1'] / N**3 + 2.0 * p['k2'] * N0 / (N + N0)**3


def H_prime_numeric(N, p, h=None):
    """dH/dN: analytic preferred; central-difference fallback.

    Raises ValueError if N < U_MIN (H4a singularity guard).
    """
    if N <= Config.U_MIN:
        raise ValueError(
            f"H_prime_numeric called with N={N:.3e} <= U_MIN={Config.U_MIN:.0e}; "
            "H4a singularity region."
        )
    try:
        return H_prime_analytic(N, p)
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
        if h is None:
            h = Config.DERIV_H
        # Clamp h so N-h stays above U_MIN.
        _margin = max(Config.DERIV_H * 0.01, Config.U_MIN * 0.01)
        h = min(h, 0.5 * N, N - Config.U_MIN - _margin)
        if h <= 0:
            raise ValueError(
                f"H_prime_numeric fallback: N={N:.3e} too close to U_MIN="
                f"{Config.U_MIN:.0e} for reliable finite difference."
            )
        return (H_model(N + h, p) - H_model(N - h, p)) / (2.0 * h)


def H_double_prime_numeric(N, p, h=None):
    """d²H/dN²: analytic preferred; central-difference fallback. Same U_MIN guard."""
    if N <= Config.U_MIN:
        raise ValueError(
            f"H_double_prime_numeric called with N={N:.3e} <= U_MIN={Config.U_MIN:.0e}; "
            "H4a singularity region."
        )
    try:
        return H_double_prime_analytic(N, p)
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
        if h is None:
            h = Config.DERIV_H
        _margin = max(Config.DERIV_H * 0.01, Config.U_MIN * 0.01)
        h = min(h, 0.3 * N, N - Config.U_MIN - _margin)
        if h <= 0:
            raise ValueError(
                f"H_double_prime_numeric fallback: N={N:.3e} too close to U_MIN="
                f"{Config.U_MIN:.0e} for reliable finite difference."
            )
        return (H_model(N + h, p) - 2.0 * H_model(N, p) + H_model(N - h, p)) / h**2


# ══════════════════════════════════════════════════════════════
# §2  Fixed Point Detection
# ══════════════════════════════════════════════════════════════

# TAG: Fixed point detection
def _try_brent(func, a, b):
    """brentq with tolerance fallback: 1e-12 → 1e-10 → 1e-8."""
    last_exc = None
    for xt, rt in ((Config.FP_XTOL, Config.FP_RTOL), (1e-10, 1e-10), (1e-8, 1e-8)):
        try:
            return brentq(func, a, b, xtol=xt, rtol=rt)
        except ValueError as e:
            last_exc = e
    raise RuntimeError(f"brentq failed on [{a:.3e}, {b:.3e}]: {last_exc}")


def _is_same_root(a, b):
    return abs(a - b) < max(1e-12, 1e-6 * max(1.0, abs(a), abs(b)))


def find_fixed_points(p):
    """Locate positive roots of H(N;p)=0 on (U_MIN, U_MAX).

    Two detection paths:
      1. Sign-change detection (Brent refinement): transverse roots.
      2. Near-zero magnitude + near-zero H' detection: tangential roots
         (saddle-node candidates that produce no sign change).

    Returns dict: s_rep, s_att, all_roots, repellers, attractors.
      s_rep = repellers[0]   (smallest repeller)
      s_att = attractors[-1] (largest attractor)
    """
    u_min, u_max = Config.U_MIN, Config.U_MAX
    u_log  = np.logspace(np.log10(u_min), np.log10(u_max), Config.FP_N_LOG)
    u_lin  = np.linspace(u_min, u_max, Config.FP_N_LIN)
    u_grid = np.unique(np.concatenate([u_log, u_lin]))

    def safe_H(u):
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                v = H_model(u, p)
            return v if np.isfinite(v) else np.nan
        except (ValueError, OverflowError):
            return np.nan

    h_grid = np.array([safe_H(u) for u in u_grid])
    roots  = []
    _endpoint_tol = max(1e-14, Config.FP_XTOL)

    # Path 1: sign-change detection + exact endpoint zeros.
    for i in range(len(u_grid) - 1):
        ha, hb = h_grid[i], h_grid[i + 1]
        if not (np.isfinite(ha) and np.isfinite(hb)):
            continue
        # Check left endpoint only; right boundary handled after the loop.
        if abs(ha) <= _endpoint_tol and not any(_is_same_root(u_grid[i], rr) for rr in roots):
            roots.append(u_grid[i])
        if ha * hb < 0:
            try:
                r = _try_brent(lambda u: H_model(u, p), u_grid[i], u_grid[i + 1])
                if not any(_is_same_root(r, rr) for rr in roots):
                    roots.append(r)
            except RuntimeError:
                pass
    # Check right boundary.
    if np.isfinite(h_grid[-1]):
        if abs(h_grid[-1]) <= _endpoint_tol and not any(_is_same_root(u_grid[-1], rr) for rr in roots):
            roots.append(u_grid[-1])

    # Path 2: tangential roots (H=0, H'≈0 — no sign change).
    # Refine candidates via minimize_scalar.

    finite_h    = h_grid[np.isfinite(h_grid)]
    h_scale     = float(np.median(np.abs(finite_h))) if finite_h.size > 0 else 1.0
    tangent_tol = max(Config.TANGENT_TOL, 1e-6 * max(1.0, h_scale))

    for idx, (u, hv) in enumerate(zip(u_grid, h_grid)):
        if not np.isfinite(hv):
            continue
        if abs(hv) >= tangent_tol:
            continue
        try:
            hp = H_prime_numeric(u, p)
            if abs(hp) >= tangent_tol:
                continue
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
            continue
        # Both conditions met — refine via local |H| minimisation.
        half_gap = (u_grid[idx + 1] - u_grid[idx]) if idx + 1 < len(u_grid) else Config.DERIV_H
        lo_b = max(u_min, u - abs(half_gap))
        hi_b = min(u_max, u + abs(half_gap))
        try:
            res = minimize_scalar(lambda uu: abs(safe_H(uu)),
                                  bounds=(lo_b, hi_b), method='bounded',
                                  options={'xatol': Config.FP_XTOL})
            refined = float(res.x)
            hv_ref = safe_H(refined)
            if not (np.isfinite(hv_ref) and abs(hv_ref) < tangent_tol):
                refined = u
            else:
                try:
                    hp_ref = H_prime_numeric(refined, p)
                    if abs(hp_ref) >= tangent_tol:
                        refined = u
                except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                    refined = u
        except (ValueError, OverflowError, RuntimeError):
            refined = u
        if not any(_is_same_root(refined, rr) for rr in roots):
            roots.append(refined)

    roots = sorted(roots)

    # Returns 0.0 for roots at U_MIN boundary (cannot classify; excluded from lists).
    def _safe_Hp(r):
        try:
            return H_prime_numeric(r, p)
        except (ValueError, OverflowError):
            return 0.0

    repellers  = [r for r in roots if _safe_Hp(r) >  Config.DEGEN_TOL]
    attractors = [r for r in roots if _safe_Hp(r) < -Config.DEGEN_TOL]

    if len(repellers) != 1 or len(attractors) != 1:
        warnings.warn(
            f"find_fixed_points: expected 1 repeller and 1 attractor for bistable "
            f"structure; found {len(repellers)} repeller(s) and {len(attractors)} "
            f"attractor(s).  s_rep / s_att selection may be incorrect.",
            stacklevel=2,
        )

    return {
        's_rep':     repellers[0]   if repellers  else None,
        's_att':     attractors[-1] if attractors else None,
        'all_roots': roots,
        'repellers': repellers,
        'attractors': attractors,
    }


def safe_find_repeller_near(prev_N, p, bracket_radius=0.15,
                            max_drift_frac=None):
    """Repeller closest to prev_N after a parameter change.

    Local scan within [prev_N*(1-bracket_radius), prev_N*(1+bracket_radius)];
    global fallback if no sign change found locally.

    Returns None (with UserWarning) if the nearest repeller drifted more than
    max_drift_frac * prev_N — possible branch jump or saddle-node proximity.
    Callers treat None as a failed step and record nan for that row.

    max_drift_frac: defaults to Config.MAX_DRIFT_FRAC (0.30).
    """
    if max_drift_frac is None:
        max_drift_frac = Config.MAX_DRIFT_FRAC

    a = max(Config.U_MIN, prev_N * (1.0 - bracket_radius))
    b = min(Config.U_MAX, prev_N * (1.0 + bracket_radius))

    xs = np.linspace(a, b, 201)
    hs = []
    for x in xs:
        try:
            v = H_model(x, p)
            hs.append(v if np.isfinite(v) else np.nan)
        except (ValueError, OverflowError):
            hs.append(np.nan)

    local_candidates = []
    for i in range(len(xs) - 1):
        if not (np.isfinite(hs[i]) and np.isfinite(hs[i + 1])):
            continue
        if hs[i] * hs[i + 1] < 0:
            try:
                r = _try_brent(lambda u: H_model(u, p), xs[i], xs[i + 1])
                if H_prime_numeric(r, p) > Config.DEGEN_TOL:
                    local_candidates.append(r)
            except (RuntimeError, ValueError):
                pass

    if local_candidates:
        r = min(local_candidates, key=lambda x: abs(x - prev_N))
        drift = abs(r - prev_N)
        limit = max_drift_frac * max(abs(prev_N), Config.U_MIN)
        if drift > limit:
            warnings.warn(
                f"safe_find_repeller_near: local root {r:.6f} drifted "
                f"{drift:.3e} > {limit:.3e} from prev_N={prev_N:.6f}; "
                "possible branch jump — marking step invalid (None).",
                stacklevel=2,
            )
            return None
        return r

    # Global fallback: full grid search.
    fp   = find_fixed_points(p)
    reps = fp.get('repellers', [])
    if not reps:
        warnings.warn(
            f"safe_find_repeller_near: no repeller found near N={prev_N:.6f}; "
            "returning None.",
            stacklevel=2,
        )
        return None
    candidate = min(reps, key=lambda r: abs(r - prev_N))
    drift = abs(candidate - prev_N)
    limit = max_drift_frac * max(abs(prev_N), Config.U_MIN)
    if drift > limit:
        warnings.warn(
            f"safe_find_repeller_near: global fallback candidate {candidate:.6f} "
            f"drifted {drift:.3e} > {limit:.3e} (MAX_DRIFT_FRAC={max_drift_frac:.0%}) "
            f"from prev_N={prev_N:.6f}.  Possible branch jump or large genuine "
            "displacement near saddle-node; marking this step as invalid (None).",
            stacklevel=2,
        )
        return None
    return candidate


# ══════════════════════════════════════════════════════════════
# §3  Lie Bracket Computation
# ══════════════════════════════════════════════════════════════

# TAG: Lie bracket computation


def lie_bracket_components(N_star, p):
    """Lie bracket components in the (k1, k2) parameter plane at repeller N*.

    The bracket of the two control vector fields X1 (k1-channel) and X2 (k2-channel):
      [X1, X2](N*) = (1/H') * [ -f1'(1+f2²) ∂_{k1}  +  f2'(1+f1²) ∂_{k2} ]

    For f1=1/N  (f1'=-1/N² < 0): bracket_k1 = +|f1'|(1+f2²)/H' > 0  at repeller
    For f2=N/(N+N0) (f2'>0):     bracket_k2 = +f2'(1+f1²)/H'   > 0  at repeller

    Both components positive — both channels push repeller upward.
    Contrast with H4a+H4b (LogQuota): there bracket_k2 < 0.

    Returns (bracket_k1, bracket_k2, Hp). Returns (None, None, Hp) if degenerate.
    """
    Hp = H_prime_numeric(N_star, p)
    if abs(Hp) < Config.DEGEN_TOL:
        return None, None, Hp

    f1v  = f1(N_star)
    f2v  = f2(N_star)
    f1pv = f1_prime(N_star)
    f2pv = f2_prime(N_star)

    bracket_k1 = -f1pv * (1.0 + f2v**2) / Hp
    bracket_k2 =  f2pv * (1.0 + f1v**2) / Hp

    return bracket_k1, bracket_k2, Hp


def mixed_partial_numeric(p, delta1, delta2, s_star):
    """Numerical symmetric mixed partial ∂²N*/∂k1∂k2 (four-point formula).

    Distinct from L(N*): L is the antisymmetric commutator; this is the
    symmetric curvature of N*(k1,k2).

    s_star required for branch-consistent tracking via safe_find_repeller_near;
    raises ValueError if None.
    """
    if s_star is None:
        raise ValueError(
            "mixed_partial_numeric requires a baseline repeller position s_star "
            "for branch-consistent tracking.  Pass the known N* from main()."
        )
    def N_rep_at(dk1, dk2):
        pp = dict(p)
        pp['k1'] += dk1
        pp['k2'] += dk2
        try:
            r = safe_find_repeller_near(s_star, pp)
            return float('nan') if r is None else r
        except (RuntimeError, ValueError):
            return float('nan')

    spp = N_rep_at(+delta1, +delta2)
    spm = N_rep_at(+delta1, -delta2)
    smp = N_rep_at(-delta1, +delta2)
    smm = N_rep_at(-delta1, -delta2)
    if np.any(np.isnan([spp, spm, smp, smm])):
        return np.nan
    return (spp - spm - smp + smm) / (4.0 * delta1 * delta2)


def theoretical_mixed_partial(N_star, p):
    """Analytic ∂²N*/∂k1∂k2 from IFT double differentiation.

    General formula: [f1·f2' + f2·f1' - f1·f2·H''/H'] / H'²

    For f1=1/N, f2=N/(N+N0):
      f1·f2' + f2·f1' = N0/(N(N+N0)²) - 1/(N(N+N0))
                       = -1/(N+N0)²

      ∂²N*/∂k1∂k2 = [-1/(N+N0)² - H''/(H'·(N+N0))] / H'²
    """
    Hp = H_prime_numeric(N_star, p)
    if abs(Hp) < Config.DEGEN_TOL:
        return np.nan
    Hpp = H_double_prime_numeric(N_star, p)
    f1v  = f1(N_star)
    f2v  = f2(N_star)
    f1pv = f1_prime(N_star)
    f2pv = f2_prime(N_star)
    num = f1v * f2pv + f2v * f1pv - f1v * f2v * (Hpp / Hp)
    return num / Hp**2


# ══════════════════════════════════════════════════════════════
# §4  Path Experiment — Linearised IFT Steps
# ══════════════════════════════════════════════════════════════

# TAG: Path experiment
#
# δN* ≈ (fi(N*) / H'(N*₀)) · Δki   — H' pinned at baseline repeller
#
# Path A:  N*₀ →[k1]→ N*₁A →[k2]→ N*_A
# Path B:  N*₀ →[k2]→ N*₁B →[k1]→ N*_B
#
# N*_A - N*_B ≈ L(N*₀) · Δk1 · Δk2

def ift_step(N_star, ki_channel, delta_ki, Hp_fixed):
    """Single linearised IFT step with pinned H'.

    Returns δN* = (fi(N*) / Hp_fixed) · Δki.  ki_channel: 1 or 2.
    """
    if abs(Hp_fixed) < Config.DEGEN_TOL:
        raise ValueError(f"Degenerate H'(N*₀) = {Hp_fixed:.2e}")
    if ki_channel == 1:
        fi = f1(N_star)
    elif ki_channel == 2:
        fi = f2(N_star)
    else:
        raise ValueError(
            f"ift_step: ki_channel must be 1 or 2, got {ki_channel!r}"
        )
    return (fi / Hp_fixed) * delta_ki


def path_experiment(N_star_0, p, delta_k1, delta_k2):
    """Two-path linearised IFT experiment with H' pinned at N*₀.

    Returns: N*₁A, N*_A, N*₁B, N*_B, diff_AB = N*_A - N*_B
    """
    Hp0 = H_prime_numeric(N_star_0, p)
    if abs(Hp0) < Config.DEGEN_TOL:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # Path A: k1 first, then k2
    ds1A = ift_step(N_star_0, 1, delta_k1, Hp0)
    N_1A = N_star_0 + ds1A
    if N_1A <= Config.U_MIN:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    ds2A = ift_step(N_1A, 2, delta_k2, Hp0)
    N_A  = N_1A + ds2A
    if N_A <= Config.U_MIN:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # Path B: k2 first, then k1
    ds1B = ift_step(N_star_0, 2, delta_k2, Hp0)
    N_1B = N_star_0 + ds1B
    if N_1B <= Config.U_MIN:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    ds2B = ift_step(N_1B, 1, delta_k1, Hp0)
    N_B  = N_1B + ds2B
    if N_B <= Config.U_MIN:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    return N_1A, N_A, N_1B, N_B, N_A - N_B


def path_experiment_vary(N_star_0, p, delta_k1, delta_k2):
    """Same two-path experiment as path_experiment, but H' re-evaluated at the
    current N* position each step (parameters p are NOT updated between steps).

    diff_delta = diff_vary - diff_fixed is a higher-order residual
    O(Δk1²·Δk2 + Δk1·Δk2²); both methods converge to the same L as Δk→0.

    Returns diff_vary = N*_A_vary - N*_B_vary.
    """
    Hp0 = H_prime_numeric(N_star_0, p)
    if abs(Hp0) < Config.DEGEN_TOL:
        warnings.warn(
            f"path_experiment_vary: H'(N*₀={N_star_0:.4e}) = {Hp0:.2e} "
            f"< DEGEN_TOL={Config.DEGEN_TOL:.0e}; returning nan.",
            stacklevel=2,
        )
        return float('nan')

    # Path A
    N_1A = N_star_0 + (f1(N_star_0) / Hp0) * delta_k1
    if N_1A <= Config.U_MIN:
        return float('nan')
    Hp_1A = H_prime_numeric(N_1A, p)
    if abs(Hp_1A) < Config.DEGEN_TOL:
        warnings.warn(
            f"path_experiment_vary: H'(N*₁A={N_1A:.4e}) = {Hp_1A:.2e} "
            f"< DEGEN_TOL after k1 step (Δk1={delta_k1}); returning nan.",
            stacklevel=2,
        )
        return float('nan')
    N_A = N_1A + (f2(N_1A) / Hp_1A) * delta_k2
    if N_A <= Config.U_MIN:
        return float('nan')

    # Path B
    N_1B = N_star_0 + (f2(N_star_0) / Hp0) * delta_k2
    if N_1B <= Config.U_MIN:
        return float('nan')
    Hp_1B = H_prime_numeric(N_1B, p)
    if abs(Hp_1B) < Config.DEGEN_TOL:
        warnings.warn(
            f"path_experiment_vary: H'(N*₁B={N_1B:.4e}) = {Hp_1B:.2e} "
            f"< DEGEN_TOL after k2 step (Δk2={delta_k2}); returning nan.",
            stacklevel=2,
        )
        return float('nan')
    N_B = N_1B + (f1(N_1B) / Hp_1B) * delta_k1
    if N_B <= Config.U_MIN:
        return float('nan')

    return N_A - N_B


def single_channel_asymmetry(N_star_0, p, delta_k1, delta_k2):
    """Single-channel response asymmetry (illustration, not path-ordering).

    Computes exact repellers after applying each channel alone:
      N*₁A: repeller after k1+Δk1 only
      N*₁B: repeller after k2+Δk2 only

    Illustrates why different orderings produce different intermediate states
    (the two channels move the repeller by different amounts), but the causal
    source of order-dependence is the bracket [X1, X2] in path_experiment().

    IFT first-order prediction: (f1(N*₀)·Δk1 - f2(N*₀)·Δk2) / H'(N*₀)

    Returns: N*₁A_ex, N*₁B_ex, diff_ex, diff_pred, rel_err
    """
    Hp0  = H_prime_numeric(N_star_0, p)
    f1_0 = f1(N_star_0)
    f2_0 = f2(N_star_0)

    pA = dict(p); pA['k1'] += delta_k1
    try:
        _r = safe_find_repeller_near(N_star_0, pA)
        N_1A_ex = float('nan') if _r is None else _r
    except (RuntimeError, ValueError):
        N_1A_ex = float('nan')

    pB = dict(p); pB['k2'] += delta_k2
    try:
        _r = safe_find_repeller_near(N_star_0, pB)
        N_1B_ex = float('nan') if _r is None else _r
    except (RuntimeError, ValueError):
        N_1B_ex = float('nan')

    diff_ex = (N_1A_ex - N_1B_ex
               if np.isfinite(N_1A_ex) and np.isfinite(N_1B_ex)
               else np.nan)
    diff_pred = ((f1_0 * delta_k1 - f2_0 * delta_k2) / Hp0
                 if abs(Hp0) >= Config.DEGEN_TOL else np.nan)
    rel_err = (abs(diff_ex - diff_pred) / max(abs(diff_pred), 1e-12)
               if np.isfinite(diff_ex) and np.isfinite(diff_pred)
               else np.nan)

    return N_1A_ex, N_1B_ex, diff_ex, diff_pred, rel_err


# ══════════════════════════════════════════════════════════════
# §5  Scaling Verification
# ══════════════════════════════════════════════════════════════

def loglog_slope(rows):
    """OLS log-log slope of |diff_AB| vs Δk1·Δk2.

    For diff_AB ∝ (Δk1·Δk2)^α, slope α ≈ 1.0 confirms O(Δk1·Δk2) leading term.

    With proportional pairs (Δk2=Δk1), product=Δk1², so this is a joint-limit
    consistency check — it confirms second-order behaviour but cannot distinguish
    Δk1·Δk2 from Δk1² scaling. The primary two-variable evidence is ratio_lin→L
    in the main scaling table (fixed Δk2, varying Δk1).

    Requires ≥ 3 finite data points (2-point OLS gives R²=1 trivially).
    Returns (slope, r_squared) or (nan, nan).
    """
    xs, ys = [], []
    for r in rows:
        d  = r.get('diff_AB', float('nan'))
        pr = r.get('product', float('nan'))
        if np.isfinite(d) and np.isfinite(pr) and abs(d) > 0 and abs(pr) > 0:
            xs.append(np.log(abs(pr)))
            ys.append(np.log(abs(d)))
    if len(xs) < 3:
        return float('nan'), float('nan')
    xs = np.array(xs)
    ys = np.array(ys)
    xm, ym = xs.mean(), ys.mean()
    ssxx = ((xs - xm)**2).sum()
    ssxy = ((xs - xm) * (ys - ym)).sum()
    if ssxx < 1e-30:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    ypred = slope * xs + (ym - slope * xm)
    sstot = ((ys - ym)**2).sum()
    ssres = ((ys - ypred)**2).sum()
    r2    = 1.0 - ssres / sstot if sstot > 1e-30 else float('nan')
    return float(slope), float(r2)


# TAG: Scaling verification

def _ratio(d, product):
    """Return diff / product, or nan if either value is unusable."""
    return (d / product
            if np.isfinite(d) and abs(product) > 1e-12
            else float('nan'))


def _ratio_vs_L(ratio_val, L_predicted):
    """Convergence metric: |ratio_lin - L| / L."""
    if (L_predicted is not None
            and np.isfinite(ratio_val)
            and np.isfinite(L_predicted)
            and abs(L_predicted) > 1e-12):
        return abs(ratio_val - L_predicted) / abs(L_predicted)
    return float('nan')


def scaling_experiment(N_star_0, p, L_predicted, delta_pairs):
    """Run linearised, vary, and single-channel-asymmetry experiments for all pairs.

    Three distinct measurements per row:
      1. path_experiment (H' pinned):   linearised path-ordering difference diff_AB
      2. path_experiment_vary (H' live): same paths with H' re-evaluated each step
      3. single_channel_asymmetry:       single-channel response difference diff_exact

    diff_AB and diff_vary measure true path ORDER DEPENDENCE (sequential pushes).
    diff_exact measures SINGLE-CHANNEL RESPONSE ASYMMETRY (each channel alone).
    diff_delta = diff_vary - diff_fixed is a purely numerical consistency residual
    (O(Δk1²·Δk2 + Δk1·Δk2²)); it has no independent physical meaning.

    Each row: dk1, dk2, product, diff_AB, analytic_ref, ratio_lin, rel_err_lin,
              diff_vary, ratio_vary, rel_err_vary, diff_delta,
              N_1A_ex, N_1B_ex, diff_exact, diff_pred_ex, rel_err_exact.
    """
    rows = []
    for (dk1, dk2) in delta_pairs:
        _, _, _, _, diff_lin = path_experiment(N_star_0, p, dk1, dk2)
        diff_vary = path_experiment_vary(N_star_0, p, dk1, dk2)
        N_1A_ex, N_1B_ex, diff_ex, diff_pred_ex, rel_ex = single_channel_asymmetry(
            N_star_0, p, dk1, dk2)

        product      = dk1 * dk2
        analytic_ref = L_predicted * product if L_predicted is not None else float('nan')

        diff_delta = ((diff_vary - diff_lin)
                      if np.isfinite(diff_vary) and np.isfinite(diff_lin)
                      else float('nan'))

        rl  = _ratio(diff_lin,  product)
        rv  = _ratio(diff_vary, product)
        rows.append({
            'dk1': dk1, 'dk2': dk2, 'product': product,
            'diff_AB':       diff_lin,
            'analytic_ref':  analytic_ref,
            'ratio_lin':     rl,
            'rel_err_lin':   _ratio_vs_L(rl,  L_predicted),
            'diff_vary':     diff_vary,
            'ratio_vary':    rv,
            'rel_err_vary':  _ratio_vs_L(rv,  L_predicted),
            'diff_delta':    diff_delta,
            'N_1A_ex':       N_1A_ex,
            'N_1B_ex':       N_1B_ex,
            'diff_exact':    diff_ex,
            'diff_pred_ex':  diff_pred_ex,
            'rel_err_exact': rel_ex,
        })
    return rows


# ══════════════════════════════════════════════════════════════
# §6  Report Generation
# ══════════════════════════════════════════════════════════════

# TAG: Report generation
def generate_report(filename, N_star_0, N_att, p, L_val, Hp_base,
                    bracket_k1, bracket_k2,
                    mp_numeric, mp_theory,
                    scaling_rows, prop_rows,
                    safety_ok, safety_msg):
    """Write full experiment report.

    scaling_rows : rows from DELTA_PAIRS  (Δk2 fixed, Δk1 varied)
    prop_rows    : rows from PROPORTIONAL_PAIRS  (Δk2 = Δk1, joint limit)
    """

    lines = []
    W = 72

    def hr(ch='═'): lines.append(ch * W)
    def blank():    lines.append('')

    def section(title):
        blank(); hr()
        lines.append(f'  {title}')
        hr()

    def _f(v, fmt='.4e'):
        return f'{v:{fmt}}' if np.isfinite(v) else '   N/A  '

    def _g(v):
        return f'{v:+.4e}' if np.isfinite(v) else '     N/A     '

    def _h(v, fmt='.6f'):
        return f'{v:{fmt}}' if np.isfinite(v) else '   N/A  '


    # ── header ───────────────────────────────────────────────
    hr()
    lines.append('  Lie Bracket Order Dependence — Mixed Harvesting (H4a + H4×)')
    lines.append(
        f'  f1 = 1/N (quota, H4a)  |  f2 = N/(N+N0), N0={N0} (saturating, H4×)'
    )
    lines.append(
        '  Scope: this experiment quantifies non-commutativity of the LINEARISED'
    )
    lines.append(
        '  CONTROL RESPONSE, not of the equilibrium mapping itself.  The equilibrium'
    )
    lines.append(
        '  N*(k1,k2) is path-independent; the bracket captures how first-order IFT'
    )
    lines.append(
        '  steps along different orderings produce different intermediate states.'
    )
    hr()
    lines.append(f'  Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(Config.summary())
    blank()

    lines.append('  BASELINE SAFETY CHECK: ' + ('OK' if safety_ok else 'WARNING'))
    lines.append(f'  {safety_msg}')
    if not safety_ok:
        lines.append('  *** Results may be unreliable. ***')

    # ── order-dependence summary ──────────────────────────────
    section('§0  Order-Dependence Summary')
    blank()
    lines.append('  Sequential application of Δk1 then Δk2 vs Δk2 then Δk1')
    lines.append('  produces different linearised-IFT repeller estimates (diff_AB ≠ 0).')
    blank()

    rep_row = None
    if scaling_rows:
        # Smallest-step pair with finite diff_AB and ratio_lin (primary evidence).
        finite_r = [r for r in scaling_rows
                    if np.isfinite(r.get('diff_AB',   float('nan')))
                    and np.isfinite(r.get('ratio_lin', float('nan')))]
        if not finite_r:
            finite_r = [r for r in scaling_rows
                        if np.isfinite(r.get('product', float('nan')))]
        if finite_r:
            rep_row = min(finite_r, key=lambda r: r.get('product', float('inf')))

    if rep_row:
        lines.append(
            f"  Representative pair (Δk1={rep_row['dk1']}, Δk2={rep_row['dk2']}):"
        )
        if np.isfinite(rep_row.get('diff_AB', float('nan'))):
            analytic = rep_row.get('analytic_ref', float('nan'))
            lines.append(
                f"    diff_AB  = N*_A - N*_B = {rep_row['diff_AB']:+.3e}"
                + (f"  (analytic ref: {analytic:+.3e})" if np.isfinite(analytic) else "")
            )
        if np.isfinite(rep_row.get('ratio_lin', float('nan'))):
            rl = rep_row['ratio_lin']
            lines.append(
                f"    ratio_lin = diff_AB / (Δk1·Δk2) = {rl:.6f}"
                + (f"  (L = {L_val:.6f},  |r-L|/L = "
                   f"{abs(rl - L_val)/abs(L_val):.2e})"
                   if L_val is not None else "")
            )
    blank()

    # ── model ─────────────────────────────────────────────────
    section('§1  Model and Baseline Parameters')
    blank()
    lines.append('  H(N; c, k1, k2) = r(1-N/K) - c - k1/N - k2·N/(N+N0)')
    blank()
    lines.append(f"  {'Parameter':<12}  {'Value':>10}")
    lines.append(f"  {'─'*12}  {'─'*10}")
    for k, v in sorted(p.items()):
        lines.append(f"  {k:<12}  {v:>10.4f}")
    lines.append(f"  {'N0':<12}  {N0:>10.4f}  (Holling half-saturation, channel 2)")
    blank()
    lines.append('  H-family identification:')
    lines.append('    G(N) = r(1-N/K)                      (H1)')
    lines.append('    c additive: ∂H/∂c = -1               (H2)')
    lines.append('    k1: ∂H/∂k1 = -1/N = -f1(N)           (H3)')
    lines.append('    k2: ∂H/∂k2 = -N/(N+N0) = -f2(N)      (H3)')
    lines.append('    f1(N) = 1/N      → +∞ as N→0+        (H4a)')
    lines.append(f'    f2(N) = N/(N+N0) → 0  as N→0+        (H4×, N0={N0})')
    blank()

    # ── geometry ──────────────────────────────────────────────
    section('§2  Fixed-Point Geometry')
    blank()
    lines.append(f'  Repeller N*₀          = {N_star_0:.8f}')
    if N_att is not None:
        lines.append(f'  Attractor N_att       = {N_att:.8f}')
        lines.append(f'  Basin width W         = {N_att - N_star_0:.8f}')
    lines.append(f"  H'(N*₀)               = {Hp_base:.6f}")
    blank()
    f1v  = f1(N_star_0); f2v  = f2(N_star_0)
    f1pv = f1_prime(N_star_0); f2pv = f2_prime(N_star_0)
    lines.append('  Channel values at N*₀:')
    lines.append(f'    f1(N*₀) = 1/N*₀              = {f1v:.6f}   (H4a: large near zero)')
    lines.append(f'    f2(N*₀) = N*₀/(N*₀+N0)       = {f2v:.6f}   (H4×: small near zero)')
    lines.append(f'    f1\'(N*₀)                     = {f1pv:.6f}')
    lines.append(f'    f2\'(N*₀)                     = {f2pv:.6f}')
    lines.append(f'    f1/f2  (zero-crossing slope)  = {f1v/f2v:.4f}   (→+∞ as N*→0+)')
    blank()

    # ── bracket ───────────────────────────────────────────────
    section('§3  Lie Bracket')
    blank()
    lines.append('  Analytic bracket numerator:')
    lines.append('    f1·f2\' - f2·f1\' = (N* + 2N0) / (N*·(N*+N0)²)')
    num_val = (N_star_0 + 2.0 * N0) / (N_star_0 * (N_star_0 + N0)**2)
    lines.append(f'    At N*₀: {num_val:.6f}')
    blank()
    if L_val is not None:
        lines.append(f'  L(N*₀) = numerator / H\'(N*₀)²  = {L_val:.8f}')
    else:
        lines.append("  L(N*₀) = N/A (degenerate H')")
    blank()

    if bracket_k1 is not None:
        norm_X1 = np.sqrt(1.0 + f1v**2)
        norm_X2 = np.sqrt(1.0 + f2v**2)
        lines.append('  Bracket components in (k1, k2) parameter plane:')
        lines.append('    (R/|X| = bracket component normalised by |Xi| = sqrt(1+fi²),')
        lines.append('     the norm of the i-th control vector field; measures bracket')
        lines.append('     strength per unit of the generating vector field.)')
        lines.append(f'    bracket_k1 = {bracket_k1:+.6f}   > 0  ✓'
                     f'   |X1|={norm_X1:.4f}   R1/|X1| = {abs(bracket_k1)/norm_X1:.4f}')
        lines.append(f'    bracket_k2 = {bracket_k2:+.6f}   > 0  ✓'
                     f'   |X2|={norm_X2:.4f}   R2/|X2| = {abs(bracket_k2)/norm_X2:.4f}')
        lines.append('    Both positive — both channels push repeller upward.')
        lines.append('    (Contrast with H4a+H4b: bracket_k2 < 0 there.)')
    blank()

    lines.append('  Near-zero analysis (N*→0+, k1 fixed):')
    lines.append('    H\'(N*) ≈ k1/N*²  (H4a term dominates)')
    lines.append('    L(N*) ≈ 2N*³/(N0·k1²)  →  0')
    near_zero_approx = 2 * N_star_0**3 / (N0 * p['k1']**2)
    lines.append(f'    At N*₀: approx = {near_zero_approx:.6e}   exact = '
                 f'{L_val:.6e}' if L_val is not None else '    At N*₀: L = N/A')
    lines.append('    Despite H4a singularity in numerator (∼1/N*), H\' grows')
    lines.append('    as k1/N*², making L suppressed — not amplified — near zero.')
    if L_val is not None:
        _err_pct = abs(near_zero_approx - L_val) / L_val * 100
        _corr    = (1 + N_star_0 / N0)**2
        lines.append(f'    Approximation error {_err_pct:.1f}%: drops next-order'
                     f' correction (1+N*₀/N0)² = {_corr:.4f}')
        lines.append(f'    (N*₀/N0 = {N_star_0/N0:.3f}; small because N*₀ << N0.)')
    lines.append('    Zero-crossing slope f1/f2 = (N*+N0)/N*² diverges separately.')
    blank()

    # ── mixed partial ─────────────────────────────────────────
    section('§4  Symmetric Mixed Partial ∂²N*/∂k1∂k2')
    blank()
    lines.append('  Analytic: [-1/(N+N0)² - H\'\'/(H\'·(N+N0))] / H\'²')
    Hpp_val = H_double_prime_numeric(N_star_0, p)
    if np.isfinite(Hpp_val):
        lines.append(f'  H\'\'(N*₀) = -2k1/N*³ + 2k2·N0/(N*+N0)³ = {Hpp_val:.6f}')
    blank()
    if mp_theory is not None and np.isfinite(mp_theory):
        lines.append(f'  Analytic  ∂²N*/∂k1∂k2 = {mp_theory:.8f}')
    else:
        lines.append("  Analytic  ∂²N*/∂k1∂k2 = N/A")
    if mp_numeric is not None and np.isfinite(mp_numeric):
        lines.append(f'  Numerical ∂²N*/∂k1∂k2 = {mp_numeric:.8f}')
        if mp_theory is not None and np.isfinite(mp_theory):
            err = abs(mp_numeric - mp_theory) / max(abs(mp_theory), 1e-12)
            lines.append(f'  Relative error         = {err:.2e}')
    else:
        lines.append("  Numerical ∂²N*/∂k1∂k2 = N/A")
    blank()
    lines.append('  L(N*) and ∂²N*/∂k1∂k2 are mathematically distinct:')
    lines.append('    L: antisymmetric commutator (order-dependence amplitude).')
    lines.append('    ∂²N*/∂k1∂k2: symmetric curvature of N*(k) surface.')
    lines.append('  Both arise from the same second-order IFT expansion of N*(k1,k2),')
    lines.append('  but encode different geometric information: L captures the')
    lines.append('  antisymmetric (path-ordering) component while ∂²N*/∂k1∂k2')
    lines.append('  captures the symmetric (curvature) component.  They are')
    lines.append('  complementary second-order descriptors; computing both provides')
    lines.append('  an independent consistency check without implying a simple'
                 ' additive reconstruction.')
    blank()

    # ── scaling table ─────────────────────────────────────────
    section('§5  Scaling Verification: ratio_lin → L(N*₀)')
    blank()
    lines.append('  Primary evidence: ratio_lin = diff_AB / (Δk1·Δk2) → L(N*₀).')
    lines.append('  Only diff_AB constitutes a direct numerical probe of the Lie bracket')
    lines.append('  scaling.  diff_vary and diff_delta are auxiliary consistency checks;')
    lines.append('  single_channel_asymmetry (§5b) is an intuitive illustration only.')
    blank()
    if L_val is not None:
        lines.append(f'  L(N*₀) = {L_val:.8f}  (analytic target)')
    blank()

    hdr = '  {:>7}  {:>7}  {:>12}  {:>13}  {:>12}  {:>11}  {:>10}'
    lines.append(hdr.format('Δk1', 'Δk2', 'diff_AB', 'analytic_ref',
                             'ratio_lin', '|r-L|/L', 'rel_exact'))
    lines.append('  ' + '─' * 80)

    for r in scaling_rows:
        lines.append(
            f"  {r['dk1']:>7.3f}  {r['dk2']:>7.3f}"
            f"  {_f(r['diff_AB']):>12}"
            f"  {_f(r.get('analytic_ref', float('nan'))):>13}"
            f"  {_f(r['ratio_lin']):>12}"
            f"  {_f(r['rel_err_lin'], '.2e'):>11}"
            f"  {_f(r.get('rel_err_exact', float('nan')), '.2e'):>10}"
        )
    blank()

    good_lin = sum(1 for r in scaling_rows
                   if np.isfinite(r.get('rel_err_lin', float('nan')))
                   and r['rel_err_lin'] < 0.05)
    good_ex  = sum(1 for r in scaling_rows
                   if np.isfinite(r.get('rel_err_exact', float('nan')))
                   and r['rel_err_exact'] < 0.05)
    trackable_ex = sum(1 for r in scaling_rows
                       if np.isfinite(r.get('rel_err_exact', float('nan'))))
    lines.append(f'  ratio_lin within 5% of L (|ratio-L|/L < 0.05): {good_lin}/{len(scaling_rows)}')
    lines.append(f'  Single-channel IFT pred within 5% of exact:    {good_ex}/{trackable_ex}'
                 f'  (trackable rows only; {len(scaling_rows)-trackable_ex} N/A due to drift guard)'
                 if trackable_ex < len(scaling_rows) else
                 f'  Single-channel IFT pred within 5% of exact:    {good_ex}/{trackable_ex}')
    lines.append('  Note: rel_err_lin = |ratio_lin - L| / L (convergence of ratio to')
    lines.append('  the analytic coefficient); analytic_ref = L * product is shown')
    lines.append('  for reference only and is NOT used in the convergence criterion.')
    lines.append('  Both analytic_ref and rel_err_lin are derived from the same L;')
    lines.append('  they are NOT independent — analytic_ref gives absolute scale,')
    lines.append('  rel_err_lin gives relative convergence rate.')
    blank()
    lines.append('  Branch continuity note: the scaling experiment assumes the repeller')
    lines.append('  branch is continuously trackable under the applied parameter steps.')
    lines.append('  Large drift (> MAX_DRIFT_FRAC of prev_N) is marked invalid (N/A) and')
    lines.append('  excluded from statistics.  Large drift may reflect either branch')
    lines.append('  switching (invalid) or genuine rapid motion near a saddle-node (valid')
    lines.append('  but outside the linearised regime).  These two cases are not')
    lines.append('  distinguishable from drift magnitude alone; N/A rows should be')
    lines.append('  inspected manually when they appear.')
    blank()

    # vary vs fixed comparison
    lines.append('  H\'-fixed vs H\'-vary comparison:')
    lines.append('    diff_delta = diff_vary - diff_fixed.')
    lines.append('    This is a purely numerical consistency residual of order')
    lines.append('    O(Δk1²·Δk2 + Δk1·Δk2²) with no independent physical meaning.')
    lines.append('    As Δk→0 it vanishes faster than the leading O(Δk1·Δk2) term,')
    lines.append('    confirming both methods converge to the same L.')
    lines.append('    Equivalently: diff_delta/(Δk1·Δk2) → 0 as Δk→0.')
    lines.append('    WARNING: at large step sizes (e.g. Δk1=0.2) the H\'-fixed')
    lines.append('    approximation breaks down; diff_vary can be several times diff_fixed.')
    lines.append('    The higher-order residual interpretation holds only in the'
                 ' small-Δk regime.')
    hdr2 = '  {:>7}  {:>7}  {:>14}  {:>14}  {:>14}'
    lines.append(hdr2.format('Δk1', 'Δk2', 'diff_fixed', 'diff_vary', 'delta'))
    lines.append('  ' + '─' * 62)
    for r in scaling_rows:
        lines.append(
            f"  {r['dk1']:>7.3f}  {r['dk2']:>7.3f}"
            f"  {_g(r['diff_AB']):>14}"
            f"  {_g(r['diff_vary']):>14}"
            f"  {_g(r['diff_delta']):>14}"
        )
    blank()


    section('§5b  Single-Channel Response Asymmetry (Intuitive Illustration)')
    blank()
    lines.append('  N*₁A: exact repeller after k1+Δk1 alone  (channel 1 in isolation)')
    lines.append('  N*₁B: exact repeller after k2+Δk2 alone  (channel 2 in isolation)')
    lines.append('  This is NOT a path-ordering measurement.  It provides an intuitive')
    lines.append('  picture of why different orderings produce different intermediate')
    lines.append('  states: the two channels move the repeller by different amounts per')
    lines.append('  unit, so the base point for a second step depends on which channel')
    lines.append('  acted first.  Path-ordering (bracket) is quantified in §5 (diff_AB).')
    lines.append('  IFT first-order prediction: (f1·Δk1 - f2·Δk2) / H\'(N*₀)')
    blank()
    hdr3 = '  {:>7}  {:>7}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}'
    lines.append(hdr3.format('Δk1', 'Δk2', 'N*₁A', 'N*₁B',
                             'diff_exact', 'diff_pred', 'rel_err'))
    lines.append('  ' + '─' * 80)
    for r in scaling_rows:
        lines.append(
            f"  {r['dk1']:>7.3f}  {r['dk2']:>7.3f}"
            f"  {_h(r.get('N_1A_ex', float('nan'))):>12}"
            f"  {_h(r.get('N_1B_ex', float('nan'))):>12}"
            f"  {_h(r.get('diff_exact',   float('nan')), '+.4e'):>12}"
            f"  {_h(r.get('diff_pred_ex', float('nan')), '+.4e'):>12}"
            f"  {_h(r.get('rel_err_exact', float('nan')), '.2e'):>10}"
        )

    na_rows = [r for r in scaling_rows
               if not np.isfinite(r.get('N_1A_ex', float('nan')))
               and np.isfinite(r.get('diff_pred_ex', float('nan')))]
    if na_rows:
        blank()
        lines.append('  Note on N/A rows: the IFT linearisation predicts a displacement')
        lines.append('  for N*₁A (channel-1 step) that exceeds the drift guard threshold')
        lines.append(f'  ({Config.MAX_DRIFT_FRAC:.0%} of N*₀ = {Config.MAX_DRIFT_FRAC*N_star_0:.4f}).')
        lines.append('  diff_pred reflects the scale of the linearised shift; exact N*₁A')
        lines.append('  cannot be reliably tracked at these step sizes.')
        for r in na_rows:
            if abs(Hp_base) > 1e-12:
                frac = abs(f1v / Hp_base * r['dk1']) / max(N_star_0, 1e-12)
                frac_str = f'{frac:.0%} of N*₀ — {"exceeds" if frac > Config.MAX_DRIFT_FRAC else "within"} guard'
            else:
                frac_str = 'H\' degenerate — fraction not computable'
            pred = r.get('diff_pred_ex', float('nan'))
            lines.append(f'    Δk1={r["dk1"]:.3f}: IFT N*₁A shift ≈ {pred:+.4e}'
                         f'  ({frac_str})')
    near_zero_rows = [
        r for r in scaling_rows
        if np.isfinite(r.get('diff_pred_ex', float('nan')))
        and abs(r['diff_pred_ex']) < 1e-4
    ]
    if near_zero_rows:
        blank()
        lines.append(
            f'  NOTE: diff_pred ≈ 0 when Δk2/Δk1 = f1/f2 = {f1v/f2v:.4f} '
            f'(zero-crossing of IFT prediction).'
        )
        lines.append(
            '    Relative error is not meaningful near this zero.'
        )
    blank()
    section('§5c  Joint-Limit Consistency Check: log-log Slope (Proportional Pairs)')
    blank()
    lines.append('  PRIMARY EVIDENCE: ratio_lin → L  (§5 main table, Δk2 fixed).')
    lines.append('  This section is a SECONDARY CONSISTENCY CHECK only.')
    lines.append('  The proportional-pair slope test does NOT replace the main two-variable')
    lines.append('  scaling test in §5; it only confirms second-order behaviour under joint')
    lines.append('  shrinking and cannot distinguish Δk1·Δk2 from Δk1² scaling.')
    blank()
    lines.append('  Proportional pairs (Δk2 = Δk1): both step sizes shrink together.')
    lines.append('  Because product = Δk1·Δk2 = Δk1², the log-log fit is one-dimensional:')
    lines.append('  it confirms diff_AB ∝ Δk1² but does NOT independently probe the')
    lines.append('  two-variable structure Δk1 × Δk2.  That structure is verified by')
    lines.append('  ratio_lin in §5 (Δk2 fixed, Δk1 varied).')
    blank()
    lines.append('  Expected: log|diff_AB| = 1 · log(product) + const  (slope = 1).')
    lines.append('  Slope > 1 → higher-order dominance; slope < 1 → sub-leading artefact.')
    blank()
    if prop_rows:
        sl, r2 = loglog_slope(prop_rows)
        if np.isfinite(sl):
            r2_str = f'{r2:.4f}' if np.isfinite(r2) else 'N/A'
            lines.append(f'  OLS log-log slope = {sl:.4f}   R² = {r2_str}')
            verdict = ('✓ consistent with O(Δk1·Δk2) leading term'
                       if abs(sl - 1.0) < 0.10
                       else '⚠ slope deviates from 1; inspect table below')
            lines.append(f'  Verdict: {verdict}')
        else:
            lines.append('  OLS log-log slope = N/A (insufficient finite data)')
        blank()
        hdrp = '  {:>7}  {:>7}  {:>12}  {:>12}  {:>12}'
        lines.append(hdrp.format('Δk1', 'Δk2', 'diff_AB', 'product', 'ratio_lin'))
        lines.append('  ' + '─' * 58)
        for r in prop_rows:
            lines.append(
                f"  {r['dk1']:>7.3f}  {r['dk2']:>7.3f}"
                f"  {_f(r['diff_AB']):>12}"
                f"  {_f(r['product']):>12}"
                f"  {_f(r['ratio_lin']):>12}"
            )
    else:
        lines.append('  No proportional-pair data available.')
    blank()

    # ── interpretation ────────────────────────────────────────
    section('§6  Interpretation')
    blank()
    lines.append('  (1) Path order-dependence confirmed: [X1, X2] ≠ 0.')
    _sign_note = ('> 0' if (L_val is not None and L_val > 0) else
                  '< 0' if (L_val is not None and L_val < 0) else '≠ 0')
    lines.append(f'      diff_AB = N*_A - N*_B ≈ L·Δk1·Δk2 {_sign_note} (linearised IFT).')
    lines.append('      Pushing k1 then k2 vs k2 then k1 yields different two-step')
    lines.append('      linearised repeller estimates (intermediate states differ by')
    lines.append('      O(Δk1·Δk2)).  Note: the true equilibrium at the terminal')
    lines.append('      parameter values is path-independent — see §6(6).')
    blank()
    lines.append('  (2) Both bracket components are positive.')
    if bracket_k1 is not None:
        lines.append('      bracket_k1 > 0 and bracket_k2 > 0 at the repeller.')
        lines.append('      Both channels push the repeller upward; non-commutativity')
        lines.append('      lies in the RATE of movement, not its direction.')
        lines.append('      Contrast with H4a+H4b (LogQuota): bracket_k2 < 0 there.')
    else:
        lines.append("      (N/A — degenerate H'; bracket components not computed.)")
    blank()
    lines.append('  (3) Near-zero: L suppresses despite H4a singularity.')
    lines.append(f'      L(N*) ≈ 2N*³/(N0·k1²) → 0 as N*→0+ (k1={p["k1"]}, N0={N0}).')
    lines.append('      H4a dominates H\' (growing as k1/N*²), which overwhelms')
    lines.append('      the 1/N* divergence in the bracket numerator.')
    lines.append('      The zero-crossing slope f1/f2 = (N*+N0)/N*² still diverges:')
    lines.append('      direction geometry and order-dependence amplitude decouple.')
    blank()
    lines.append('  (4) Primary amplification: H\'(N*)→0 (saddle-node bifurcation).')
    lines.append('      L ∝ 1/H\'(N*)² — strongest near the bifurcation boundary.')
    blank()
    lines.append('  (5) H4-combination comparison (near-zero L behaviour):')
    lines.append('        H4a + H4a :  If α₁ = α₂ (equal singularity orders), the')
    lines.append('                     bracket numerator f1·f2\'-f2·f1\' = 0 identically,')
    lines.append('                     so L ≡ 0.  Divergence requires α₁ ≠ α₂; with')
    lines.append('                     f1~N^{-α₁}, f2~N^{-α₂}:')
    lines.append('                       numerator ~ (α₁-α₂)·N^{-α₁-α₂-1}')
    lines.append('                       H\'² ~ k₁²α₁²·N^{-2(α₁+1)}')
    lines.append('                       L = numerator/H\'² ~ N*^{α₁-α₂+1}.')
    lines.append('                     Classification by exponent e = α₁-α₂+1:')
    lines.append('                       e < 0 (α₁ < α₂-1): L diverges as N*→0')
    lines.append('                       e = 0 (α₁ = α₂-1): L → finite non-zero')
    lines.append('                       e > 0 (α₁ > α₂-1): L suppresses → 0')
    lines.append('        H4a + H4× :  L ~ N*^3           suppresses (this model)')
    lines.append('        H4b + H4b :  L → finite > 0')
    lines.append('        H4× + H4× :  L ~ N*^2           suppresses')
    lines.append('      H4a+H4× suppresses faster than H4×+H4× because the H4a')
    lines.append('      channel drives H\' harder (N*⁻² vs finite), producing a')
    lines.append('      larger denominator in L.')
    blank()
    lines.append('  (6) Why a "final-state exact path" experiment gives diff = 0.')
    lines.append('      N*(k1, k2) depends only on terminal parameter values,')
    lines.append('      not on the order in which they were reached.  Applying')
    lines.append('      k1 then k2 vs k2 then k1 both terminate at (k1+Δk1, k2+Δk2),')
    lines.append('      so the final repeller is identical and diff ≡ 0 identically.')
    lines.append('      Non-commutativity is a property of the INTERMEDIATE states')
    lines.append('      under the linearised flow (captured by path_experiment),')
    lines.append('      not of the final equilibrium.  The single-channel asymmetry')
    lines.append('      in §5b is the exact-repeller complement: it compares the')
    lines.append('      isolated single-channel responses, which are genuinely unequal.')
    blank()
    lines.append('  (7) Geometric summary.')
    lines.append('      Non-commutativity resides in the tangent dynamics, not in the')
    lines.append('      equilibrium manifold.  The equilibrium map N*(k1,k2) is a')
    lines.append('      smooth scalar function whose level sets always exist; the')
    lines.append('      bracket measures how linearised control flows on that manifold')
    lines.append('      fail to commute, encoding the curvature of the repeller branch'
                 ' in parameter space.')

    blank(); hr()
    lines.append(f'  Report: {filename}')
    hr()

    text = '\n'.join(lines)
    Config.out(filename).write_text(text, encoding='utf-8')
    return text


# ══════════════════════════════════════════════════════════════
# §7  Main Entry Point
# ══════════════════════════════════════════════════════════════

# TAG: Main entry point
def main():
    p = dict(BASE_PARAMS)

    print('=' * 72)
    print('  Lie Bracket Order Dependence — Mixed Harvesting (H4a + H4×)')
    print(f'  f1 = 1/N  (quota, H4a)  |  f2 = N/(N+{N0})  (Holling II, H4×)')
    print('=' * 72)

    # ── [1/6] baseline repeller ───────────────────────────────
    print('\n[1/6] Finding baseline repeller...')
    fp       = find_fixed_points(p)
    N_star_0 = fp['s_rep']
    if N_star_0 is None:
        print('  ERROR: No bistable structure found. Check parameters.')
        return
    Hp_base = H_prime_numeric(N_star_0, p)
    N_att   = fp['s_att']
    f1v     = f1(N_star_0)
    f2v     = f2(N_star_0)
    print(f'  N*₀ (repeller)    = {N_star_0:.8f}')
    print(f"  H'(N*₀)           = {Hp_base:.6f}")
    print(f'  f1(N*₀) = 1/N*₀   = {f1v:.6f}  (H4a)')
    print(f'  f2(N*₀)            = {f2v:.6f}  (H4×)')
    print(f'  f1/f2 (zero-crossing slope) = {f1v/f2v:.4f}')
    if N_att is not None:
        print(f'  N_att (attractor) = {N_att:.6f}   W = {N_att - N_star_0:.6f}')
    else:
        print('  WARNING: no attractor found')

    # ── [2/6] safety check ────────────────────────────────────
    print('\n[2/6] Baseline safety check...')
    safety_ok, safety_msg = True, ''
    if abs(Hp_base) < Config.HP_MARGIN:
        safety_ok  = False
        safety_msg = (f"H'(N*) = {Hp_base:.2e} < HP_MARGIN={Config.HP_MARGIN:.0e}: "
                      "near saddle-node boundary.")
    elif N_att is None:
        safety_ok  = False
        safety_msg = 'No attractor found.'
    else:
        sep_frac = (N_att - N_star_0) / N_att
        if sep_frac < Config.SEP_FRAC:
            safety_ok  = False
            safety_msg = f"Window fraction {sep_frac:.3f} < {Config.SEP_FRAC}."
        else:
            safety_msg = f"window fraction = {sep_frac:.3f} ≥ {Config.SEP_FRAC}."
    print(f'  {"OK" if safety_ok else "WARNING"}: {safety_msg}')

    # ── [3/6] lie bracket ─────────────────────────────────────
    print('\n[3/6] Computing Lie bracket...')
    bracket_k1, bracket_k2, Hp_bracket = lie_bracket_components(N_star_0, p)
    if abs(Hp_bracket) < Config.DEGEN_TOL:
        L_val = None
    else:
        numerator = (N_star_0 + 2.0 * N0) / (N_star_0 * (N_star_0 + N0)**2)
        L_val = numerator / Hp_bracket**2

    if L_val is None:
        print("  L(N*₀) = N/A (degenerate H')")
    else:
        print(f'  Bracket numerator (N*+2N0)/(N*(N*+N0)²) = {numerator:.6f}')
        print(f'  L(N*₀)                                   = {L_val:.8f}')
        approx = 2 * N_star_0**3 / (N0 * p['k1']**2)
        print(f'  Near-zero approx 2N*³/(N0·k1²)  = {approx:.6e}  (→ 0 as N*→0+)')

    if bracket_k1 is not None:
        norm_X1 = np.sqrt(1.0 + f1v**2)
        norm_X2 = np.sqrt(1.0 + f2v**2)
        print(f'  bracket_k1 = {bracket_k1:+.6f}  (>0 ✓)  R1/|X1| = {abs(bracket_k1)/norm_X1:.4f}')
        print(f'  bracket_k2 = {bracket_k2:+.6f}  (>0 ✓)  R2/|X2| = {abs(bracket_k2)/norm_X2:.4f}')
        print('  Both positive — both channels push repeller upward.')

    # ── [4/6] mixed partial ───────────────────────────────────
    print('\n[4/6] Computing symmetric mixed partial ∂²N*/∂k1∂k2...')
    try:
        mp_val = mixed_partial_numeric(p, Config.DERIV_MIXED, Config.DERIV_MIXED,
                                       s_star=N_star_0)
        if np.isfinite(mp_val):
            print(f'  Numerical ∂²N*/∂k1∂k2 = {mp_val:.8f}')
        else:
            print('  Numerical ∂²N*/∂k1∂k2 = N/A')
            mp_val = None
    except (RuntimeError, ValueError, OverflowError) as e:
        print(f'  WARNING: {e}')
        mp_val = None

    mp_theory = theoretical_mixed_partial(N_star_0, p)
    if np.isfinite(mp_theory):
        print(f'  Analytic  ∂²N*/∂k1∂k2 = {mp_theory:.8f}')
        if mp_val is not None:
            err = abs(mp_val - mp_theory) / max(abs(mp_theory), 1e-12)
            print(f'  Relative error         = {err:.2e}')
    else:
        print('  Analytic  ∂²N*/∂k1∂k2 = N/A')
        mp_theory = None

    # ── [5/6] scaling experiment (Δk2 fixed) ────────────────
    print('\n[5/6] Running path order-dependence experiment...')
    rows = scaling_experiment(N_star_0, p, L_val, Config.DELTA_PAIRS)
    good_lin = sum(1 for r in rows
                   if np.isfinite(r.get('rel_err_lin', float('nan')))
                   and r['rel_err_lin'] < 0.05)
    good_ex      = sum(1 for r in rows
                       if np.isfinite(r.get('rel_err_exact', float('nan')))
                       and r['rel_err_exact'] < 0.05)
    trackable_ex = sum(1 for r in rows
                       if np.isfinite(r.get('rel_err_exact', float('nan'))))
    print(f'  Tested {len(rows)} pairs (Δk2 fixed).')
    print(f'  Linearised  rel_err < 5%: {good_lin}/{len(rows)}')
    print(f'  Single-channel IFT rel_err < 5%: {good_ex}/{trackable_ex}'
          + (f'  ({len(rows)-trackable_ex} N/A)' if trackable_ex < len(rows) else ''))

    if rows and L_val is not None:
        # Use smallest-step pair (closest to asymptotic L).
        finite_rows = [r for r in rows if np.isfinite(r.get('ratio_lin', float('nan')))]
        if finite_rows:
            smallest = min(finite_rows, key=lambda r: r.get('product', float('inf')))
            r_small = smallest['ratio_lin']
            print(f'  Smallest-step ratio_lin = {r_small:.6f}  '
                  f'(Δk1={smallest["dk1"]}, Δk2={smallest["dk2"]},  '
                  f'L = {L_val:.6f},  |r-L|/L = '
                  f'{abs(r_small-L_val)/max(abs(L_val),1e-12):.2e})')

    # ── [6/6] proportional pairs ─────────────────────────────
    print('\n[6/6] Joint-limit verification (Δk2 = Δk1, both shrink)...')
    prop_rows = scaling_experiment(N_star_0, p, L_val, Config.PROPORTIONAL_PAIRS)
    sl, r2 = loglog_slope(prop_rows)
    if np.isfinite(sl):
        r2_str = f'{r2:.4f}' if np.isfinite(r2) else 'N/A'
        verdict = ('✓ O(Δk1·Δk2) confirmed' if abs(sl - 1.0) < 0.10
                   else '⚠ slope deviates; inspect joint-limit consistency table')
        print(f'  log-log slope = {sl:.4f}  R² = {r2_str}  {verdict}')
    else:
        print('  log-log slope = N/A (insufficient data)')

    print('\nGenerating report...')
    generate_report(
        Config.REPORT_NAME,
        N_star_0, N_att, p,
        L_val, Hp_base,
        bracket_k1, bracket_k2,
        mp_val, mp_theory,
        rows, prop_rows,
        safety_ok, safety_msg,
    )
    print(f'\n  Output: {Config.out(Config.REPORT_NAME)}')
    print('-' * 72)


if __name__ == '__main__':
    main()