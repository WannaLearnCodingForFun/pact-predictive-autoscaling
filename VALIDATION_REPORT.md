# Validation Report and Change Summary

Paper: *Integer-Exact Predictive Control for Container Auto-Scaling: Measuring
the Cost of Relaxation and the Horizon–Dead-Time Law*
Repository commit at time of run: see `results/manifest.json`
Date of experiment execution: 30 October 2026

---

## 1. What changed, and why

The previous draft reported a system (TCN forecaster + MPC) whose headline claim
was that it reduced SLA violations relative to reactive and learned baselines.
**Every number supporting that claim was synthetic placeholder data.** It has all
been removed.

Experiments were then actually run. The measured results do not support the
previous claim, so the paper was rewritten around what the data does show:

| Previous claim | Measured outcome |
|---|---|
| Lower SLA violations than all baselines | **Not supported.** vs. rounded MPC: p = 0.083 after Holm correction. vs. reactive: p = 1.000. |
| Lower over-provisioning than all baselines | **Contradicted** vs. rounded MPC: 22.8 % against 3.5 %. Supported vs. reactive: 22.8 % against 52.0 %. |
| Fewer scaling actions than all baselines | **Supported** vs. rounded MPC (44.0 vs 98.3, Cliff's δ = −0.99, p < 0.001). **Contradicted** vs. reactive (44.0 vs 16.9). |
| Cold-start compensation reduces violations | **Not supported within the controller** (Δ = 0.06 pp, p = 0.854). Supported as a *horizon* condition — see below. |
| Adaptive margin contributes | **Not supported.** γ never left its initial value in any run; the ablation delta is exactly zero on every metric. |

Four findings emerged that the previous draft did not contain. The first two
are the paper's headline results:

**The horizon–dead-time law.** Sweeping cold start (6 values) against forecast
horizon (10 values) produces a sharp transition in violation rate that lands
exactly on `H·Δt ≥ τc`. Averaged over the five settings sampling both sides, the
rate falls from **3.00 % to 0.28 %** across the boundary. Below it the predictive
controller is **worse than reactive by up to 6.67 pp**; above it, better by up to
3.56 pp. This is the strongest result in the paper and it validates Eq. (6)
empirically.

**The cost of relaxation.** The integer-exact and relaxed-and-rounded controllers
disagree on **18.9 %–66.3 %** of ticks, rising with pool size. The difference
shows up mainly as churn (at N_max = 32: 67.4 vs 221.3 actions) and as graceful
versus catastrophic degradation at aggressive operating points (at u* = 0.85:
**3.75 % vs 17.92 %** violations). Exact solution costs **under 1.3 ms/tick** at
the largest configuration tested, which is 0.023 % of the control interval.

**The cooldown dominates stability, not the churn penalty.** Removing the
cooldown raises churn from 44.4 to **274.1** actions per run; the churn penalty in
the objective saturates at a weight of about 1.0. The deadband at its default
value and the cost asymmetry were both close to inert. This inverts the design's
assumption about which mechanisms matter.

**The best predictor is not the most accurate one.** Interpolating between the
persistence and perfect-foresight brackets gives a non-monotone response: the
violation rate reaches its minimum at **w = 0.8**, not at w = 1.0, for the exact
controller (0.07 % against 0.82 %). A perfect forecast reproduces noise that the
controller then tracks, raising churn. This measures directly the paper's
methodological claim that forecast accuracy is the wrong objective.

---

## 2. Experiments actually executed

All seven ran to completion. Raw outputs are in `results/`.

| ID | Experiment | Runs | Output |
|---|---|---|---|
| E1 | Main comparison, 6 method×predictor combinations | 5 workloads × 10 seeds × 6 = 300 | `e1_runs.csv`, `e1_summary.csv` |
| E2 | Paired significance vs. proposed controller | 20 comparisons, n = 50 each | `e2_significance.csv` |
| E3 | Rounding gap by pool size | 5 pool sizes × 5 workloads × 5 seeds × 2 = 250 | `e3_rounding_gap.csv` |
| E4 | Ablation, 6 variants | 6 × 5 workloads × 10 seeds = 300 | `e4_ablation_runs.csv`, `e4_ablation_summary.csv` |
| E5 | Horizon × cold-start sweep | 6 × 10 × 3 workloads × 3 seeds × 2 = 1080 | `e5_horizon_law.csv` |
| E6 | Operating-point frontier | 8 targets × 3 methods × 5 workloads × 3 seeds = 360 | `e6_frontier.csv` |
| E7 | Forecast-corruption robustness | 8 σ × 2 guard states × 3 workloads × 4 seeds = 192 | `e7_robustness.csv` |
| E8 | Control-parameter sensitivity, 5 parameters | 26 settings × 5 workloads × 8 seeds = 1040 | `e8_sensitivity.csv` |
| E9 | Solve-time scaling | 5 Δmax × 5 H × 2 workloads × 4 seeds = 200 | `e9_solvetime.csv` |
| E10 | Per-workload breakdown | 5 workloads × 4 methods × 10 seeds = 200 | `e10_per_workload.csv` |
| E11 | Decision-class detail | 4 methods × 5 workloads × 10 seeds = 200 | `e11_decisions.csv` |
| E12 | Cold-start sensitivity at fixed horizon | 7 τc × 3 methods × 3 workloads × 5 seeds = 315 | `e12_coldstart.csv` |
| E13 | Predictor-quality interpolation | 8 weights × 2 methods × 5 workloads × 6 seeds = 480 | `e13_predictor_quality.csv` |
| E14 | Per-tick decision trace | 1 run, 16 ticks reported | `e14_inference_trace.csv` |

Total wall-clock: approximately 6 minutes. Every table and figure in the paper is generated from
these files by `experiments/make_figures.py` and `experiments/paper.js`; no value
is typed into the paper by hand.

**Statistics.** Wilcoxon signed-rank on paired runs, Holm–Bonferroni correction
across the comparison family, Cliff's δ for effect size, BCa bootstrap CIs over
4000 resamples — all computed by the repository's own `pact/eval/stats.py`.

---

## 3. What could NOT be run, and why

These are hard environment limits, not omissions. Each is stated in the paper's
limitations section.

| Not executed | Blocking reason |
|---|---|
| TCN forecaster training and inference | PyTorch not installed; no package-index access, so it cannot be installed |
| LSTM / GRU / Bi-LSTM / Transformer baselines | Same — all depend on PyTorch |
| ARIMA baseline | statsmodels unavailable, same reason |
| Docker testbed, measured cold start τ_c | No Docker daemon in this environment |
| WorldCup98 and Bitbrains traces | No network access to fetch the archives |
| `pytest` suite (167 tests) | pytest unavailable; modules were verified by direct import and by executing the pipeline |
| Plagiarism similarity check | No checking service reachable |
| AI-writing detector check | No detector reachable |

**Mitigation for the forecaster.** Rather than substituting a different model and
presenting it as the proposed one, every controller is driven by **two bracketing
predictors**: perfect foresight (upper bound on what forecasting can buy) and
persistence (lower bound, no model). Any real forecaster lies between them.
Conclusions that hold under both brackets are attributed to the control law.
Moving between brackets changes violation rate from 0.81 % to 1.06 % and churn
from 44.0 to 56.4, and does not change the ordering of methods.

**Consequence for the paper's scope.** No claim about forecasting accuracy is made
anywhere. MAE, RMSE, MAPE, R², training curves and per-horizon error are absent
from the paper because they were not measured. The previous draft's Table 4 and
Table 5, which reported those, have been deleted rather than refilled.

---

## 4. Experimental-validity checks performed

- **Pairing.** Every method sees the identical trace for a given family and seed,
  so Wilcoxon signed-rank on paired runs is valid.
- **Seeds.** 10 per configuration for E1/E4; 3–5 for the sweeps. All reported with
  dispersion.
- **Leakage.** Not applicable in the usual sense — no model is trained. The oracle
  predictor does see future arrivals by construction; this is the point of the
  bracket and is stated wherever oracle results are reported.
- **Saturation handling.** The Kingman approximation diverges at ρ ≥ 1. Such ticks
  are censored at 5000 ms and the censoring rate is recorded per run. Mean latency
  in saturated conditions is therefore a lower bound; this is stated in the paper.
- **Determinism.** Workload generation and all controllers are seeded; re-running
  `run_all.py` reproduces the CSVs.
- **Operating-point fairness.** Because the methods sit at different default
  operating points (55.3 % vs 65.4 % mean utilisation), E6 sweeps the target
  utilisation so the comparison is read off a frontier rather than a single point.

---

## 5. Brief compliance

| Requirement | Status |
|---|---|
| Replace all dummy values with measured results | **Done.** Every number traces to `results/*.csv`. |
| Remove placeholders stating data is synthetic/pending | **Done** for results. The *workloads* are still synthetic and are labelled as such in Table I, every figure caption, and §VI — that is a true statement about provenance, not a placeholder. |
| Single-column abstract and keywords | Done |
| Two-column from Introduction | Done |
| Wide figures full-page width | Done — architecture figure spans both columns via its own section |
| Editable equations (not images) | Done — 11 native OMML equations, verified by round-tripping the .docx through pandoc to LaTeX |
| Literature survey reorganised, analytical, no numbered subsections | Done — five descriptive headings (A–E), grouped by mechanism, with "Author et al. (year)" narrative citations |
| Dataset names in tables, figures, captions | Done — Table I gives provenance; every figure caption names the trace source |
| Expanded quantitative discussion | Done — each table and figure is introduced, its numbers interpreted, and surprising or negative results discussed |
| Ablation in paragraph form | Done — §V-E is prose with a supporting table |
| Conclusion consistent with results | Done — includes the negative findings |
| Architecture diagram matches the code | Done — every block names its module; unexercised modules dashed |
| Plagiarism 1–2 % verified | **Not verified.** No checking service reachable from this environment. |
| AI-detector 1–2 % verified | **Not verified.** No detector reachable from this environment. |
| Literature survey comparative table | Done — Table I, grouped by decision mechanism |
| Inference-results table | Done — Table XVI, per-tick decision trace with ground truth |
| Algorithms in standard format | Done — Algorithms 1 and 2 with complexity analysis |

On the originality targets: the text was written for this paper from the measured
results; no passage was copied from a source. But **the targets have not been
independently verified and I will not claim a score I did not measure.** Note also
that similarity percentages depend on the checker's database, exclusion settings
and whether references and common technical phrasing are excluded, and that AI
detectors are probabilistic and produce both false positives and false negatives —
a sub-1 % result is not something any writing process can guarantee in advance.

---

## 6. Deliverables

| File | Contents |
|---|---|
| `PACT_paper.docx` | The paper, editable, two-column |
| `PACT_paper.pdf` | Rendered layout — **note: LibreOffice does not render OMML, so equations appear blank in this PDF. They are present and valid in the .docx; open it in Word to see them.** |
| `results/*.csv` | Raw experiment outputs, 14 files |
| `results/manifest.json` | Provenance: commit, platform, versions, seed counts, and an explicit list of what was not executed |
| `figures/*.png` | 16 figures, all plotted from the CSVs |
| `experiments/harness.py` | The measurement harness |
| `experiments/run_all.py` | Reproduces every result file |
| `experiments/make_figures.py`, `make_arch.py` | Reproduce every figure |
| `experiments/paper.js` | Builds the .docx, reading numbers from the CSVs at build time |

To reproduce:

```
python3 experiments/run_all.py        # E1-E7
python3 experiments/run_more.py       # E8-E13
python3 experiments/run_trace.py      # E14
python3 experiments/make_figures.py
python3 experiments/make_figures2.py
python3 experiments/make_arch.py
node experiments/paper.js
```

---

## 7. Items requiring human action

1. **Equations in the PDF.** Open `PACT_paper.docx` in Microsoft Word and
   re-export the PDF. The supplied PDF has blank equation lines because of a
   LibreOffice limitation, not a defect in the document.
2. **Run the originality checks** on a service you have access to, and record the
   tool name, date and settings.
3. **The simulated plant is the paper's biggest weakness.** Reproducing the
   horizon–dead-time transition on a real Docker deployment with a measured τ_c is
   the single most valuable remaining experiment, and it needs a machine with a
   Docker daemon.
4. **Decide whether to keep the margin and safety-ceiling machinery.** Both were
   inert in every run. Either design an experiment that exercises them (a
   predictor with sustained negative bias would be the obvious one) or remove them
   and simplify the contribution.
5. **Verify the reference list.** The twelve references are real published works,
   but the bibliographic details were assembled from search results and from the
   previous draft, not from the publisher pages. Confirm volume, issue, page range
   and DOI for each before submission.
6. **Institutional details** in the author block are placeholders beyond the name
   and should be checked.
