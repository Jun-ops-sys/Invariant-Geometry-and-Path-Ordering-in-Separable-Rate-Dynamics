# Invariant Geometry and Path-Ordering in Separable Rate Dynamics
## Computational code

Two independent scripts. Run them in any order.

---

### CVE_Controlled_Experiments.py

Numerical validation for §4 (Sections 4.1–4.2, Tables 1–2).

Two controlled-variable experiments:

| Experiment | Fixed | Varied | Tests |
|---|---|---|---|
| CVE-I | f(s) = 1/s (H4a) | G(s) | Reachability as independent switch |
| CVE-II | G(s), all parameters | f(s) | H4 condition as independent switch |

**Output** (written to working directory):

| File | Content |
|---|---|
| `allee_ratio_audit.csv` | CVE-I arm A (Allee+Quota) |
| `clark76_ratio_audit.csv` | CVE-I arm B (Clark 1976) |
| `specialist_ratio_audit.csv` | CVE-II arm A (Specialist, H4✗) |
| `generalist_ratio_audit.csv` | CVE-II arm B (Generalist, H4b) |
| `cve_ii_summary.csv` | CVE-II cross-arm summary |
| `cve_report.txt` | Full text report |

The four `*_ratio_audit.csv` files share the same columns:
`k_val, c_val, s_rep, s_att, W, is_bistable, f_theory, r_num, r_fd, r_ift, abs_err, rel_err, dH_ds, used_ift, orth_r_analytic, orth_r_numeric, orth_rel_err, orth_theta`

Full description of numerical methods in Appendix B of the paper.

---

### Lie_Bracket_Mixed_Harvesting.py

Numerical validation for §5–6 (Table 3, §6.2).

Model: H(N; c, k₁, k₂) = r(1 − N/K) − c − k₁/N − k₂·N/(N+N₀)

Computes and verifies:
- Lie bracket components [X₁,X₂]^{k₁} and [X₁,X₂]^{k₂}
- Path-ordering coefficient L(N\*)
- Mixed partial ∂²N\*/∂k₁∂k₂ (symmetric component)
- Path experiments (Table 3): convergence of ratio → L as Δk → 0

Output is written to `lie_bracket_mixed_harvesting_report.txt` and printed to stdout. Full description of numerical methods in Appendix C of the paper.

---

### Requirements

```
numpy>=1.21
scipy>=1.7
pandas>=1.3
```

Python 3.8+.

---

### License

MIT License. See `LICENSE` for details.