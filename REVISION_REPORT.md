# PACT — Revision Report

Covers the seven-phase revision of *Integer-Exact Predictive Control for
Container Auto-Scaling: Measuring the Cost of Relaxation and the
Horizon–Dead-Time Law* (`PACT_paper.docx`) and the repository behind it.

The original manuscript and the results it was built from are preserved:

| Preserved artefact | Location |
|---|---|
| Results from the superseded solver | `results_greedy_backup/` |
| Superseded solver (still used as a comparison arm) | `pact/control/mpc.py` |
| Manuscript before this revision | git history, tag/commit prior to this change |

---

## 1. The finding that drove the revision

The reviewer's objection to Algorithm 2 was correct, and it was more serious
than a wording problem.

`pact/control/mpc.py::solve_mpc` enumerates the admissible first actions and
then completes each horizon with `_greedy_plan`, which tracks demand one step
at a time. The manuscript called the resulting first action *exact*, on the
grounds that only the first action is applied. That argument does not hold:
the first action is chosen by comparing the costs of **completed** plans, so a
biased completion biases the comparison. The word "Integer-Exact" is in the
paper's title, so this was not a detail.

Two things made the bias concrete rather than theoretical.

**The greedy completion zeroes the tail.** `_greedy_u` returns 0 once the
landing index passes the end of the horizon, so every plan is costed as if all
late actions were zero. With `r = 0.15` and `s = 0.60`, an all-zero tail after
`u = 4` costs `s·(0−4)² = 9.60`, while a ramp-down `4 → 2 → 1 → 0` costs
`4.35`. The completion therefore overstates the cost of any plan ending on a
large action, and the overstatement depends on the plan, so it does not cancel
in the argmin.

**The error is large enough to change decisions.** Measured in
`experiments/verify_exact.py` — see §3.

### What replaced it

`pact/control/mpc_exact.py::solve_mpc_exact` solves the same integer program
to global optimality. It rests on two structural properties of the dead-time
plant, both of which are stated in the paper and tested in the repository:

1. **Dead-time truncation.** `u(t+j)` does not appear in `N` until tick
   `t+j+τ`, so it can enter a tracking term only when `j + τ ≤ k + 1`. Over
   `k = 0 … H−1` that confines the tracking-relevant actions to
   `m = max(1, H − τ + 1)`. At `H = 12, τ = 9` this is `m = 4`: 9⁴ = 6 561
   sequences instead of 9¹² ≈ 2.8 × 10¹¹.
2. **Tail separability.** The remaining actions land after the horizon, so
   they enter the cost only through the decoupled chain
   `Σ[r u² + s(u − u_prev)²]` plus their own admissibility — a scalar
   shortest-path problem solved exactly by backward recursion.

Saturation of `N` at the pool bounds does not weaken either argument, because
saturation depends only on what has already landed. This mattered in practice:
an initial version computed the trajectory from a closed-form cumulative sum
and was wrong by 152.0 cost units on a case where the `n_min` clamp bound.
The shipped version simulates the real plant and is checked against it.

The recursion is memoised on the full controllable state
`(k, N, pending, u_prev)`. That covers the opposite regime too: when `τ` is
small the pending queue is short, so distinct prefixes collapse onto the same
state and the solve is polynomial. The case `τ = 1, H = 12, Δmax = 4`, which
plain enumeration cannot touch at 9¹², solves in **6.0 ms**.

---

## 2. Where the exact solver is used, and where it is not

Between the two regimes there is a band — dead time long enough to keep the
queue long, horizon long enough to make `m` large — in which neither reduction
bites. `exact_is_tractable()` detects it from `(H, τ, Δmax)` and the harness
falls back to the greedy completion there, recording which solver each run used
(`exact_solver` in `summarise()`, and `mpc_solver` in `results/manifest.json`).

- Every result at the paper's configuration (`H = 12, τ = 9`) is **exact**.
- The horizon sweep (E5) is the one experiment that crosses the band. The
  exact count of cells solved exactly is recorded per cell in
  `results/e5_horizon_law.csv` (`exact_frac`) and quoted in the manuscript
  from that column rather than stated here.
- Every cell immediately either side of the `H·Δt = τc` boundary is solved
  exactly, so the transition the paper measures is not affected by the
  fallback; the fallbacks sit well above the boundary, where the controller
  is already at its violation floor.

A first version of `exact_is_tractable()` got this wrong in the optimistic
direction: it assumed memoisation always collapses the search once
`m > τ − 1`, when in fact the memo still indexes the pending queue, so it
collapses nothing while the queue is longer than the search is deep. That
version declared intractable configurations tractable and the solver ran for
minutes on a single tick. The predicate now bounds both the enumeration and
the memo, and — more importantly — `solve_mpc_exact` carries a hard node cap
and raises `ExactBudgetExceeded` rather than returning a truncated search as
though it were the optimum. A truncated search that silently passed for an
exact one would be the same class of error this revision exists to fix.

This is disclosed in the manuscript rather than smoothed over.

---

## 3. Numerical verification of the solver

`experiments/verify_exact.py`, results in `results/e15_validation.json` and
`results/e15_exact_vs_greedy.json`.

### Part 1 — against exhaustive enumeration

Ten configurations small enough to enumerate the whole `(2Δmax+1)^H` space,
40 random plant states each, every candidate scored with the same reference
objective (`plan_cost`, which replicates the repository cost term by term and
adds the landing-admissibility test).

| Quantity | Value |
|---|---|
| Plant states brute-forced | 400 |
| `solve_mpc_exact` attained the brute-force optimum | **400 / 400** |
| Greedy completion returned an optimal first action | **354 / 400 (88.5 %)** |
| Greedy objective excess, mean / max | 7.97 / 192.80 |

### Part 2 — at the paper's configuration (τ = 9, H = 12, m = 4)

| Quantity | Value |
|---|---|
| Plant states compared | 1 200 |
| First actions in agreement | 1 070 (89.17 %) |
| **First actions differing** | **130 (10.83 %)** |
| Greedy objective strictly worse | 540 (45.00 %) |
| Greedy objective strictly better | **0** |
| Mean / max relative objective excess | 1.148 % / 99.53 % |
| Exact solve, median / p95 / max | 28.34 / 40.94 / 90.92 ms |
| Greedy solve, median / p95 / max | 0.269 / 0.397 / 1.469 ms |

The gap is one-sided — the greedy completion never beats the exact solver,
which is the expected signature of an exact reference. It is also
concentrated rather than uniform: mean relative excess 1.1 %, maximum 99.5 %.
A controller built on it is almost always identical and occasionally
substantially wrong.

Exactness costs about 105× the compute. In absolute terms the worst observed
solve was 90.9 ms, **1.82 % of the 5 000 ms control interval**.

---

## 4. Reference audit

Every reference was checked against a primary source. Two were fabricated and
two were misattributed.

| # | Status | Action |
|---|---|---|
| [1] Van Do et al., HPA properties | Title and authorship corroborated; **venue could not be confirmed** — the only locatable copy is a preprint | Cited as a preprint; flagged below |
| [2] Jeong & Jeong, ARAScaler | **Verified** — TSC 18(1):72–84 | DOI `10.1109/TSC.2024.3522815` added |
| [3] Fog et al., SOSE 2025 | **Verified** — pp. 151–161 | DOI `10.1109/SOSE67019.2025.00022` added |
| [4] "Baresi, Guinea, Leva, Quattrocchi", Euro-Par 2018 | **MISATTRIBUTED** — the paper is by **E. Incerto, M. Tribastone, C. Trubiani** | Authors corrected; LNCS vol. 11014 and DOI `10.1007/978-3-319-96983-1_11` added; the in-text "Baresi et al. (2018)" corrected to "Incerto et al. (2018)" |
| [5] Quattrocchi et al., TSC 2024 | **Verified** — 17(3):804–820 | DOI `10.1109/TSC.2024.3354062` added |
| [6] "D. Zhang et al.", OptScaler | **MISATTRIBUTED** — the paper is by **Ding Zou et al.** | Author list corrected in full; DOI `10.14778/3685800.3685829` added; in-text "Zhang et al. (2024)" corrected to "Zou et al. (2024)" |
| [7] Cai et al., TSC 2025 | Existence corroborated; co-authors unverified | Reduced to "Z. Cai et al." |
| [8] Meng et al., CATScaler | Exists (IEEE Xplore 11095836); volume/issue/pages unverified | Reduced to "F. Meng et al."; unverified volume/issue/pages removed |
| [9] "Ding et al., Informer, IEEE Access 13:14572–14583" | **NO EVIDENCE FOUND** after repeated search | **Removed** |
| [10] Fang, Ma, Chen, Buyya, HGraphScale | Authors verified from the author manuscript; volume/issue/pages unverified | Kept; unverified "vol. 19, no. 1, pp. 410–422, 2026" removed |
| [11] "Habibi, Rawat, Leon-Garcia, AgileSphere, IEEE Access 14:129559–129579" | **NO EVIDENCE FOUND** after two extended searches | **Removed** |
| [12] Kumar, Verma, Buyya, arXiv:2606.07046 | **Verified** | Kept |

The reference list is now 10 entries. Table I lost the two rows for the
removed references and the remaining entries were renumbered, as were all
in-text citations.

**On similarity checking:** no Turnitin or iThenticate report was obtained,
because neither service is reachable from this environment. No plagiarism
percentage is claimed anywhere in the manuscript or here. What was done is a
source-verification exercise, which is a different thing and is reported as
such.

---

## 5. Manuscript corrections

### Mathematical and algorithmic

- **Algorithm 2 replaced.** The pseudocode is now the memoised exact
  recursion. The false justification ("exact … because the receding-horizon
  principle discards the rest of the plan") is gone, replaced by the
  truncation and separability arguments with the numbers that verify them.
- **Complexity restated.** `Θ((2Δmax+1)·H)` was wrong for the old algorithm
  and is wrong for the new one. It is now `O((2Δmax+1)^m · H)` in the
  long-dead-time regime and polynomial in the short one.
- **Section V-I added** reporting the exact-versus-greedy comparison.
- **Abstract, introduction and conclusion** restructured around three
  findings rather than two, with the tractability result stated as its own
  contribution and the solver cost reported honestly (it rose by two orders
  of magnitude and is still under 2 % of the control interval).

### Factual and internal consistency

- **Five-versus-six cold-start settings resolved.** The sweep covers six
  values of `τc`, but at `τc = 5 s` every horizon in the grid already
  satisfies `HΔt ≥ τc`, so that setting yields no below-boundary measurement.
  The average is over the remaining **five**. The abstract and conclusion
  already computed this from the data (`cliffDrop.length`); Section V-A
  hard-coded "six" and was wrong. It now derives the count from the data and
  explains the exclusion.
- **Duplicated sections removed.** `inferenceTrace`, `positioning` and
  `discussion` were each spliced into the document **twice**. Fixed.
- **Section lettering repaired.** Section V had two subsections lettered "F"
  and no "M"; letters are now A–P with no gaps or collisions, and all
  cross-references were updated to match.
- **Table numbering repaired.** Two tables were numbered XVII. All 18 tables
  are now sequential in document order, with cross-references updated.

### Writing

Formulaic constructions were removed by rewriting the surrounding argument,
not by substituting synonyms:

| Removed | Replaced with |
|---|---|
| "Three observations follow from the table and motivate what is measured in Section V." | An opening that states the finding itself ("Nowhere in the table is the replica count treated as an integer decision variable…") |
| "The honest reading of this experiment is that neither controller dominates." | "Neither controller dominates." — the hedge added nothing |
| "This is the clearest result in the paper and it is the one with the most direct practical use." | A sentence that says *why* it is useful: the condition can be checked before deployment from one latency measurement |
| "The table makes the contribution and its boundary explicit in the same view." | A direct statement of which three columns differ and which row runs the other way |

---

## 6. Document rendering

Three defects made the PDF materially worse than the source.

- **Tables overflowed their frame.** The two-column text block is 10 080
  twips wide, so a body column is 4 840 (3.36 in). Tables were declared at
  8 100–11 700 twips — up to 2.4× the column — and were clipped at the page
  edge (Table VIII lost its last column entirely). `mkTable` now fits every
  table to its frame and centres it, and tables of four or more columns are
  emitted in a one-column continuous section spanning both body columns,
  carrying their caption with them. This is the standard IEEE treatment.
- **Six figures were authored at 6.7–6.8 in and displayed at 3.35 in**,
  halving every axis label and legend to 3–4 pt. They now span both columns
  at their authored size. Raster resolution was raised from 400 to 600 dpi,
  the architecture figure was redrawn at a size closer to how it is shown,
  and PDF export no longer downsamples images (`experiments/topdf.sh`).
- **Equations rendered blank.** The eleven display equations were native
  OMML, which is valid and editable in Word but is silently dropped by
  LibreOffice — and therefore by the exported PDF. They are now rendered at
  600 dpi from `experiments/make_equations.py`, whose transcription of each
  equation sits next to the equation number it carries. The trade is
  deliberate and worth stating: the equations are no longer editable in
  Word, but they are visible everywhere. The OMML builder (`EQ` in
  `build_paper.js`) is retained if that trade is not wanted.

---

## 7. Test suite

`pytest` cannot be installed here (no package-index access), so
`experiments/run_tests.py` runs the suite with a thin stand-in
(`experiments/_pytest_shim.py`) supplying `raises`, `approx`, `fixture`,
`mark`, `importorskip` and `parametrize` expansion.

| | |
|---|---|
| Test files | 23 |
| Passed | **67** |
| Failed | **0** |
| Skipped (need real fixtures) | 18 |
| Import errors | 9 |

Eight import errors are `torch`, which is unavailable. The ninth was a
genuine syntax error in the repository's own test file —
`tests/test_eval_manifest.py` line 23 read `path.unlink():` — now fixed.

`tests/test_mpc_exact.py` is new: it brute-forces the full decision space on
seven configurations spanning both dead-time regimes and checks that the
exact solver attains the optimum, that the tail optimum is not all-zero, and
that the exact objective never exceeds the greedy one.

---

## 8. Unresolved limitations

These are unchanged by this revision or newly introduced by it, and none is
hidden in the manuscript.

1. **No real deployment.** The plant is `pact.sim.queue_model.PoolSimulator`,
   an analytical M/G/c model with a Kingman p95 approximation. No Docker
   daemon is available, so the actuation backend was never exercised and the
   cold-start interval is configured, not measured.
2. **No trained forecaster.** `torch` cannot be installed, so the TCN was
   never run. Every controller is bracketed between perfect foresight and
   persistence. Conclusions holding under both brackets are attributable to
   the control law; nothing here establishes what a trained predictor would
   do.
3. **Synthetic workloads only.** The five trace families are generated. The
   public archives could not be fetched.
4. **The exact solve is not universally tractable.** It is cheap at long dead
   time relative to the horizon, and cheap at short dead time through state
   collapse. In the band between, it is not, and the controller falls back.
   §2 gives the measured extent.
5. **Reference [1]'s venue is unconfirmed.** The work exists and the
   authorship is corroborated, but no published venue could be located; it is
   cited as a preprint.
6. **References [7], [8] and [10] carry reduced detail** because co-author
   lists or volume/issue/pages could not be verified against a primary
   source. Nothing unverified was retained.
7. **No similarity report.** See §4.
8. **The margin and safety ceiling remain inert** across every experiment.
   They are reported as inert rather than removed, because that is the
   measurement.
9. **Single service.** One replica pool; no inter-service coupling.
