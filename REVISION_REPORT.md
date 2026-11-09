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

- Every result at the paper's configuration (`H = 12, τ = 9`) is **exact**:
  `exact_tick_pct` is 100 in every PACT run of `e1_runs.csv`,
  `e4_ablation_runs.csv` (the `no_mpc` variant has no optimiser, hence 0) and
  `e17_trained_runs.csv`.
- Three experiments vary `H` or `τc` far enough to cross the band:
  - **E5**, the synthetic horizon sweep: 14 of its 60 cells fall back
    (`results/e5_horizon_law.csv`, `exact_frac`), at τc = 30 s for H ≥ 12,
    45 s for H ≥ 16, 60 s for H ≥ 20 and 90 s for H ≥ 24. All 14 are above
    the `H·Δt = τc` boundary, and the first cell above the boundary is exact
    at every τc, so the transition the paper measures is not affected.
  - **E18**, the same grid on the two recorded families: the same 14
    (τc, H) cells fall back on each, 28 of 120 (`results/e18_horizon_law_real.csv`).
  - **E12**, the cold-start sweep at fixed `H = 30`: the predictive
    controller is exact at τc = 5 s and 15 s (`exact_tick_pct` = 100) and
    greedy on every tick at τc ≥ 30 s. Behavioural columns of the re-run
    match the previous `e12_coldstart.csv`. Table XIII and Fig. 14 mark
    the greedy rows.
- The E9 solve-time grid records the solver per cell; `H = 20` and `H = 30`
  are greedy at every `Δmax`, so E9's timings for those rows are timings of the
  completion, not of the exact solve.

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
| Exact solve, median / p95 / max | 15.17 / 19.66 / 34.25 ms |
| Greedy solve, median / p95 / max | 0.273 / 0.339 / 0.903 ms |

The gap is one-sided — the greedy completion never beats the exact solver,
which is the expected signature of an exact reference. It is also
concentrated rather than uniform: mean relative excess 1.1 %, maximum 99.5 %.
A controller built on it is almost always identical and occasionally
substantially wrong.

Exactness costs about 56× the compute at the median. In absolute terms the
worst observed solve was 34.3 ms, **0.69 % of the 5 000 ms control
interval**. These timings are from a run on an otherwise idle machine; an
earlier run made while other experiments were executing measured 1.9–2.7×
longer (28.3 / 40.9 / 90.9 ms) with identical decisions, so the
absolute timings in the manuscript are the idle-machine ones and are labelled
as such.

---

### Cost of exactness, and one optimisation

The exact solve is between one and two orders of magnitude more work per
tick than the greedy completion (56× at the median, 38× at the maximum). That is the honest price of the claim, and it is
reported rather than buried; what matters is that the absolute cost stays a
small fraction of the control interval.

One round of profiling was done, because the re-run cost scales with it.
The inner loop is the zero-tail rollout, which runs once per enumerated
prefix — 512k calls per 120 solves at the paper's configuration. It was
rebuilding an eight-element tuple and re-summing the pending queue at every
step, and reconstructing a saturation-correction vector that is all zeros
whenever nothing saturates, which is nearly always. With every tail action
zero the queue simply drains, so the item landing at stage `m+j` is
`pipe[j]`; walking that directly removes the tuple churn and the repeated
sums, and sharing one all-zero correction tuple lets every unsaturated
prefix share a single tail memo. A 400-tick closed-loop run went from 12.9 s
to 8.7 s, median solve 20.8 ms.

The six exactness tests were re-run against brute force after the change and
still pass. That is the reason for having them: a performance rewrite of a
solver whose whole claim is optimality is exactly where a silent regression
would hurt most, and the brute-force comparison is what rules it out.

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
  findings rather than two (four after the recorded-traffic work of §9), with the tractability result stated as its own
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
  and no "M"; letters were made gap- and collision-free, and all
  cross-references were updated to match. The recorded-traffic results of §9
  add Section V-Q.
- **Table numbering repaired.** Two tables were numbered XVII. Tables are now
  sequential in document order (23 after §9), with cross-references updated.

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

Seven defects made the PDF materially worse than the source. The first
three were found in the first pass; the remaining four surfaced only once
the pages were rasterised and read, which is the check that should have
come first.

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
- **Every image was 25 % too small.** `docx` sizes images in pixels at
  96 dpi; the builder passed `inches × 72`, so each figure and equation
  rendered at 72/96 of its intended width. The architecture figure asked
  for 6.95 in and was placed at 5.21 in; the display equations landed
  around 13 pt tall beside 9.5 pt body text, which is why they read as
  grey smudges rather than mathematics. One constant, but it affected
  every image in the document.
- **Figure label sizes were inconsistent between figures.** No plot called
  `tight_layout`, so the default subplot margins left a wide white border
  that `bbox_inches="tight"` then cropped. Each PNG therefore came out far
  smaller than its `figsize` and had to be scaled up on the page — by
  1.02× to 1.83×, depending on how much white the crop removed. A label
  specified at 8 pt thus landed anywhere between 8 and 15 pt. Adding
  `tight_layout` and standardising every figure to one of the two frames
  the paper uses (3.35 in in-column, 6.90 in full-width) brings all
  seventeen to 1.00× at ~597 effective dpi.
- **Captions were separated from what they caption.** Column balancing is
  free to place a figure in one column and its caption in the next, and
  did: Table III's caption was stranded alone at the top of a page, and
  Fig. 2 was dealt into the left column with its caption in the right.
  Table rows are now `cantSplit` with `keepNext`, and each in-column
  figure is bound to its caption inside one unbreakable block. A related
  bug: a wide table opening its one-column section pulled in the caption
  immediately before it, which was usually the caption of the figure
  above — that rule now checks what the caption belongs to.
- **Algorithm blocks wrapped and split.** The pseudocode runs to about 65
  monospace characters, which does not fit a 3.36 in column at a readable
  size, so comment tails such as `// Eq. (1)` fell onto lines of their own
  and Algorithm 2 broke across two pages. Both algorithms now span the
  text block and cannot split.
- **The architecture figure's labels overflowed their boxes.** Box
  geometry is in data units and labels are in points, so the two scale
  differently with `figsize` and the labels collided with neighbouring
  boxes. `make_arch.py` now measures each label against the box holding it
  and shrinks any that overruns, which keeps the diagram correct at
  whatever width it is rendered.

After these, a sweep of the 33 rendered pages of the current PDF finds no
content outside the side margins (≈0.75 in) or below ≈0.49 in of the
bottom edge (the footer). Some full-width figures open a page with a
large top gap; captions stay with their tables and figures.

---

## 7. Test suite

Two runners are used. `pytest` is available in the project venv.

| Runner | Files | Passed | Failed | Skipped |
|---|---|---|---|---|
| `pytest -q` | 27 test modules | **186** | **0** | 1 |
| `experiments/run_tests.py` | 24 files it enumerates | **145** | **0** | 42 (fixture-gated) |

`tests/test_mpc_exact.py` brute-forces the full decision space on seven
configurations spanning both dead-time regimes and checks that the exact
solver attains the optimum, that the tail optimum is not all-zero, and
that the exact objective never exceeds the greedy one.

`ruff` and `mypy` are clean on `pact/`, `tests/` and `scripts/` (the
configured scope). `experiments/` is outside that scope; newly added
files there (`run_real.py`, `train_forecasters.py`, `make_real_traces.py`,
`lambda_forecast.py`, `make_fig_real.py`, `run_stage.py`) are ruff-clean
as well.

`experiments/verify_numbers.py` locates 35 of 35 checked values in the
rendered document after the final build.

---

## 7b. Final state of the result files

Every experiment E1–E13 and E15 was re-run under `solve_mpc_exact` and
under the start-state rule of §9. E3, E9 and E15 were then re-run again
on an otherwise idle machine; behavioural columns were identical to the
loaded copies (kept in `/tmp/pact_loaded/`) and the idle-machine
timings are the ones quoted in the manuscript:

| Experiment | Quantity | Idle-machine value |
|---|---|---|
| E15 | exact p50 / p95 / max | 15.17 / 19.66 / 34.25 ms |
| E3 | max mean solve | 17.54 ms |
| E9 | default (Δmax = 4, H = 12) mean | 10.42 ms |
| E9 | Δmax = 8, H = 12 mean | 56.16 ms |

Two findings came out of the exact-solver re-run rather than out of the
solver change itself, and both are in the manuscript:

- **A tuning conclusion was an artifact.** Under the greedy completion,
  raising the churn weight to unity drove the violation rate to 0.0089 %
  and held it there at s = 1.0, 1.5 and 2.0. Under the exact solver the
  same sweep sits near 0.38–0.42 % (`results/e8_solver_compare.csv`,
  Section V-I). The direction of the effect survives (over-provisioning
  36.9 → 22.2 %, oscillations 52.6 → 33.7 as s goes 0 → 1); the
  magnitude did not.
- **The rate limit stops binding.** Δmax = 4, 6 and 8 give identical
  service level, capacity and action counts in `e8_sensitivity.csv`.
  Median solve time at H = 12 nonetheless rises 10.42 → 56.16 ms
  (`e9_solvetime.csv`). Raising Δmax past about four buys identical
  behaviour at five times the compute.

Claims that survived the re-run (worth recording because the temptation
is to report only what moved):

- The horizon–dead-time transition, at every synthetic cold-start
  setting that straddles the boundary.
- The integrality gap at aggressive operating points (16.18 % rounded
  against 2.86 % exact at u\* = 0.85).
- The absence of a significant service-level advantage at the default
  operating point: pact against rounded on violations, p_holm = 0.103,
  still not significant.
- The cooldown as the dominant stability control.

Claims that **did not** survive, and were corrected rather than
retained:

- The adaptive margin and safety ceiling are **not** inert. The ceiling
  issued 140 overrides in the E1 exact-oracle runs and 160 in the E4
  full-controller synthetic runs; the margin moved in 3 of 20 recorded
  oracle runs in E4 and in 63–86 of 86 recorded runs under imperfect
  predictors (E17, E22). An earlier manuscript called both inert; the
  HEAD result files already showed the ceiling firing and did not
  record `gamma_max` at all, so the margin claim had no measurement
  behind it.
- Removing the cold-start term is **not** a no-op: it cuts
  over-provisioning by 16.9 points and churn by 19.9 actions (E4
  `no_coldstart`).
- The exact-versus-reactive gap is **not** "three orders of magnitude"
  everywhere; the E3 violation gap range is computed from the file.
- The unsourced Δmax timings 20.8 / 58.8 / 110.9 / 204 ms were replaced
  by E9 cells at H = 12.
- Cooldown is not free; asymmetry is not inert; the churn curve is not
  flat.

E12 was re-run once more after `exact_tick_pct` was added to its
writer. Behavioural columns matched the previous file; the new column
shows exact solution only at τc = 5 s and 15 s.

`results/manifest.json` names the solver, lists the executed files
including E16–E22, and keeps a `not_executed` block. It currently
records `git_dirty: true` on commit `59f4ec0523`; the SHA on the
remote is that commit, and the uncommitted work is this revision. The
manifest should be regenerated after a commit and push.

---

## 8. Unresolved limitations

None of these is hidden in the manuscript.

1. **No real deployment.** The plant is
   `pact.sim.queue_model.PoolSimulator`, an analytical M/G/c model with
   a Kingman p95 approximation. No Docker daemon is available, so the
   actuation backend was never exercised and the cold-start interval is
   configured, not measured.
2. **One trained forecaster, one seed.** Torch is installed and a TCN
   was trained (188 504 parameters, 17 epochs, MPS). Every neural
   baseline was also trained once. There is no training-variance
   estimate. The TCN ranked 6 of 7 on nMAE at the 12-step horizon
   (6.70 % against 5.62 % for ARIMA).
3. **One recorded trace, rescaled.** Five synthetic families plus two
   scalings of WorldCup98 days 41–50. Bitbrains GWA-T-12 was located
   and verified (`data/provenance.json`) but not used: it is 5-minute
   per-VM utilisation, not request arrivals.
4. **The exact solve is not universally tractable.** Cheap at long dead
   time relative to the horizon, and cheap at short dead time through
   state collapse. In the band between, the controller falls back.
   §2 gives the measured extent (E5: 14/60; E18: 28/120; E12: 5 of 7
   cold-start settings).
5. **Reference [1]'s venue is unconfirmed.** Cited as a preprint.
6. **References [7], [8] and [9] carry reduced detail** because
   co-author lists or volume/issue/pages could not be verified against
   a primary source.
7. **No similarity report.** See §4. No plagiarism percentage is
   claimed.
8. **The margin and safety ceiling have unestablished value.** They
   are no longer described as inert. Disabling either produced no
   significant change in the violation rate (E4 under the oracle; E22
   under persistence and the TCN). E22's full-controller violation is
   at most 0.006 %, so those tests have almost no room to show a
   service-level benefit. The ceiling issued 0 overrides on recorded
   traffic. The corruption model is symmetric, so it never produces
   the sustained under-prediction the ceiling exists to catch.
9. **Single service.** One replica pool; no inter-service coupling.
10. **The horizon–dead-time law was not testable on recorded traffic.**
    The reactive baseline never violated on either WorldCup98 scaling
    (E18).
11. **The cause of the trained arm's churn reduction is unidentified.**
    E21 shows the TCN's mean bias accounts for the capacity-side
    departures (14 of 14) but agrees in direction on only 4 of 8
    churn-side comparisons. The smoothing hypothesis from E13 was not
    tested.

---

## 9. Recorded traffic and the trained forecaster

This is the work after the exact-solver revision. Every experiment,
synthetic and recorded, was re-run under the start-state rule
`initial_replicas = max(4, min(ceil(λ[0]/(μ u*)), n_max))` after E20
showed that a fixed start of four replicas put every violation on
WC98/test in the first 30 ticks (92 early ticks falling to 0). The
rule changes 5 of 45 synthetic windows (diurnal) and 9 of 9 WC98/test
windows.

### Data

WorldCup98 days 41–50 (5–14 June 1998), 41 files, 250 257 178 requests,
fetched 2026-11-05. Provenance, per-file SHA-256 and the ITA copyright
notice are in `data/`. Two scalings: WC98/trace (whole-trace peak at
85 % of pool capacity) and WC98/test (test-split peak at 85 %). The
test split is 13–14 June, a weekend.

### E16 — forecast accuracy

At H = 12, nMAE: ARIMA 5.62 %, Transformer 5.75 %, GRU 5.81 %, BiLSTM
6.08 %, LSTM 6.41 %, TCN 6.70 %, persistence 6.72 %. The TCN's bias at
H = 12 is +2.64 req/s and it under-forecasts on 26.1 % of windows.

### E17 — trained arm versus the brackets

24 comparisons (2 families × 2 controllers × 6 metrics). Placement:

| Position | Count |
|---|---|
| Between the brackets | 7 (1 degenerate: WC98/test, exact, violation, all 0) |
| Better than both | 13 |
| Worse than both | 4 |

The trained arm did **not** land between the brackets per metric. The
outside-the-bracket comparisons were reported as measured; the model
was not re-tuned. 18 differences vs persistence are significant after
Holm correction (14 in the trained arm's favour, 4 against); 18 vs the
oracle.

### E21 — bias probe

The oracle was shifted by the TCN's mean bias (2.145 req/s on
WC98/trace, 8.647 on WC98/test). The shifted oracle matches the trained
arm's direction in 14 of 14 capacity-side comparisons and is closer to
it in 14 of 14. Churn-side: 4 of 8 agree. The rounded controller's
churn halving under the trained arm is not explained by bias.

### E22 — margin and ceiling under imperfect predictors

The ceiling issued 0 overrides, so disabling it is identical. The
margin moved in 63–86 of 86 runs. Disabling it never changed
violations significantly (floor effect: full-controller violation ≤
0.006 %) and significantly lowered over-provisioning in 3 of 4 cases
(−1.29, −2.43, −1.15 pp).

### E18 / E19 — law and ceiling on recorded traffic

E18: reactive never violates on either family. WC98/test is 0
everywhere; WC98/trace has 1–4 violation ticks of 3600, only above the
boundary at long H. Verdict: not testable. E19: 0 ceiling overrides;
curves coincide.

### Other corrections made while reading the rendered text

- Section IV lettering had two "F" subsections and a skipped "B"; it
  is now A–G.
- `experiments/build_paper.js` CSV loader did not honour RFC-4180
  quotes, so Table XXI read the wrong margin counts from `e1_runs.csv`
  (whose label field contains commas). Fixed with `splitCsvLine`.
- Empty Holm p-values (identical paired runs) were counting as
  significant under `Number('') < 0.05`. Guarded; Table XX now reads
  18 vs persistence, not 19.
- V-I claimed the horizon sweep was the one experiment that crossed
  the fallback band; E12 and E18 also do. Disclosed.
- V-J called the H = 30 sweep "integer-exact" on every row; only τc =
  5 s and 15 s are.
- Configuration guidance had claimed deadband and asymmetry "neither
  was found to matter"; both claims were already contradicted by
  Table X and are now computed from E8.
- The conclusion's "firmest of the three results" sentence now
  restricts the horizon law to the synthetic families.

The controller remains `pact/control/mpc_exact.py`. The greedy
completion in `pact/control/mpc.py` is used only as the E15 comparison
arm and as the recorded fallback.
