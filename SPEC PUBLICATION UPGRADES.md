# Publication-Grade Upgrades

What separates the current repo from a paper a reviewer at a real venue would
accept. Companion to `CURSOR_SPEC.md`.

**Read this first.** The implementation is in good shape. What is missing is not
code quality — it is the evidence a reviewer demands before believing a claim.
Roughly in order of how badly each gap would hurt in review:

| # | Gap | Severity | Mostly code? |
|---|---|---|---|
| U1 | No statistical testing of differences | Fatal | Yes |
| U2 | No published method reimplemented as a baseline | Fatal | Yes |
| U3 | No stability / feasibility analysis of the MPC loop | Severe at a control venue | No — this is maths |
| U4 | No robustness study under forecast degradation | Severe | Yes |
| U5 | Single service pool only | Severe | Yes, large |
| U6 | No controller overhead measurement | Moderate | Yes, small |
| U7 | Narrow workload coverage | Moderate | Yes |
| U8 | No reproducibility artifact | Moderate | Yes, small |
| U9 | No threats-to-validity or negative results | Moderate | No — writing |

U1, U2, U6, U8 are a week of work and lift the paper from "course project" to
"submittable". U3 and U5 are what make it *good*. Be realistic about which you
are going for.

---

## U1 — Statistical testing

Right now you report mean ± std over 5 runs. A reviewer asks: is the difference
between PACT and the LSTM baseline actually significant, or is it noise?

Build `pact/eval/stats.py`:

- **Paired comparison across seeds.** Same seeds, same traces, both methods →
  paired samples. Use the Wilcoxon signed-rank test (non-parametric; n=5 is far
  too small to assume normality).
- **Bootstrap confidence intervals** on every headline metric. 10,000
  resamples, BCa if practical, percentile otherwise. Report 95% CI, not just std.
- **Effect size.** Cliff's delta for the non-parametric case. A statistically
  significant 0.2% improvement in SLA violation is not an interesting result, and
  reporting effect size is how you show you know the difference.
- **Multiple-comparison correction.** You compare across 5+ methods and 3
  datasets. Holm–Bonferroni over the family of comparisons.
- Raise `n_seeds` from 5 to **at least 10**. With 5, Wilcoxon cannot reach
  p < 0.05 at all for some configurations — the test is underpowered by
  construction, and a reviewer who knows this will notice immediately.

Every results table gains a CI column and a significance marker. Add a short
"Statistical methodology" paragraph to §4.1 stating the tests, the seed count,
and the correction — reviewers look for this paragraph and its absence is read
as carelessness.

---

## U2 — Reimplement a published baseline

This is the single most common reason an autoscaling paper is rejected:
*"compared only against trivial baselines."*

Threshold HPA, ARIMA and a plain LSTM are all strawmen in a 2026 venue. You need
at least one **published predictive autoscaler**, reimplemented faithfully, from
your own literature review. Pick one or two:

- A queueing-theoretic reactive controller in the style of QN-CTRL (ref [4]).
  Strong, fast, and the closest existing work to yours — beating it is the
  result that matters. If you cannot beat it, that is itself a publishable
  finding, reported honestly.
- A Transformer/Informer-based predictive scaler in the style of refs [16]/[17].
- An RL-based scaler in the style of Gwydion (ref [11]), via `gym-hpa` if the
  environment is usable.

Build these under `pact/baselines/published/`, one module each, with a docstring
citing the paper and stating explicitly **what you changed and why** — you will
not reproduce anything exactly, and saying so is far better than implying you did.

Add a "Baseline fidelity" paragraph to §4.4 listing each deviation. Reviewers
forgive approximations that are disclosed; they do not forgive ones they discover.

---

## U3 — Stability and feasibility analysis

This is what makes the paper *technical* rather than empirical, and it is the
upgrade that most changes which venues will take it.

You have an MPC controller with dead time, integer decisions and hard
constraints. The standard questions, none of which the paper currently answers:

1. **Recursive feasibility.** If the optimisation is feasible at tick `t`, is it
   feasible at `t+1`? With `n_max` as a hard ceiling and demand that can exceed
   what `n_max` serves, it is **not** — and your Algorithm 1 already silently
   saturates. Say so, characterise the infeasible set, and state what the
   controller does there (saturate at `n_max` and let the safety ceiling handle
   it). A disclosed limitation is a contribution; an undisclosed one is a hole.

2. **Closed-loop stability.** Under bounded forecast error `|e| ≤ ε`, does `N(t)`
   converge to a neighbourhood of the true requirement, and how large is that
   neighbourhood? For a linear plant with dead time and a quadratic cost this is
   tractable. The integer rounding and the asymmetric weight make it messy —
   handle it by analysing the relaxed system and bounding the rounding error by 1.

3. **Worst-case SLA bound.** The headline theorem: given forecast error bounded
   by ε, cold start τ, and a maximum demand slew rate, bound the worst-case
   duration of under-provisioning. This connects your design parameters to the
   quantity the operator cares about, and it is the kind of result that gets a
   paper into a control or systems venue rather than a workshop.

4. **Margin loop interaction.** γ is an integral controller wrapped around a
   loop that already has an integral-like character. Two integrators can
   oscillate. Show the timescale separation (κ_γ small, γ moving over minutes vs
   control over seconds) makes this safe — or find that it doesn't and add the
   separation explicitly.

Then **verify each claim numerically.** `pact/eval/stability.py` sweeps forecast
error magnitude, measures the realised deviation, and plots it against the
derived bound. A theorem with a matching empirical curve is far more convincing
than either alone.

This is the one item here that is genuinely hard and mostly not code. If your
Process Control background is going to pay off anywhere in this project, it is
here.

---

## U4 — Robustness under forecast degradation

Your whole thesis is that forecasting helps. The obvious attack is: what happens
when it doesn't work?

`pact/eval/robustness.py` — inject controlled corruption into the forecast and
measure degradation:

- Additive Gaussian noise at increasing σ
- Systematic bias, positive and negative (the asymmetric loss should make
  negative bias much worse — show it)
- Phase lag: shift the forecast by k ticks
- Total failure: forecast frozen at the last value (predictor crashed)
- Distribution shift: train on D1, test on D2 with no retraining

Headline result: **the degradation curve**. Plot SLA violation against forecast
corruption, with the reactive baseline as a horizontal line. The crossing point
— where PACT becomes *worse* than plain reactive scaling — is the most honest
and most interesting number in the paper. Find it and report it.

This also justifies the safety ceiling empirically rather than by assertion: run
the same sweep with the ceiling disabled and show where it saves you.

---

## U5 — Multi-service coupled scaling

Your current system scales one pool. Every real deployment has many, and they
interact through shared hosts and call graphs. A reviewer will ask, and
"future work" is a weak answer for something this central.

Minimum viable version:

- Two or three services in a call chain (frontend → backend → cache)
- Per-service forecasters, one shared MPC over the joint replica vector
- Coupling through a shared host-capacity constraint:
  `Σ_i N_i · c_i ≤ C_host` added to the constraint set in Eq. 23
- Propagated demand: the backend's arrival rate is a function of the frontend's
  served rate, so the forecast must flow down the chain

This is a substantial extension — realistically several weeks. If you cannot do
it, the fallback is a short §4.7 reporting what happens when you run *independent*
PACT instances per service and showing the interference, which at least
demonstrates you understand the problem.

---

## U6 — Controller overhead

Cheap to measure, and reviewers ask every time. `pact/eval/overhead.py`:

- Controller CPU and RSS as a percentage of total testbed consumption
- Per-tick latency breakdown: scrape, feature build, inference, mapping, MPC
  solve, actuation — p50/p95/p99 each
- Model size on disk, parameter count, quantised vs float
- Tick-deadline miss rate

Add a table to §4.1. Your claim is that this runs as a sidecar on modest
hardware; this is the evidence for it, and without numbers it is an assertion.

---

## U7 — Workload coverage

Three traces, one of them synthetic, is thin. Add at least one more real trace
with different character:

- Azure Functions trace (bursty, serverless, very different shape)
- Google cluster trace 2019 (large-scale, heterogeneous)
- Alibaba 2018 or 2020 (microservice call graphs, and directly comparable to
  several papers in your own review)

Alibaba is the highest-value addition: several works you cite use it, so it lets
you position your numbers against published ones.

Also make the **burst rise-time sweep** a headline figure, not a buried
appendix. Sweep rise time from instant to several minutes and plot the
predictive advantage against it. The advantage should be large for fast bursts
and vanish for slow ones. Showing exactly where your method stops helping is a
strength, and reviewers read it as confidence rather than weakness.

---

## U8 — Reproducibility artifact

- `make reproduce` — one command, from clean checkout to every table and figure
- Lockfile (`uv.lock` or `requirements.txt` with hashes). Pin everything.
- Seed manifest: `results/manifest.json` recording git SHA, config hash, seeds,
  package versions, host specs, and τc for every run
- A reproduction Dockerfile so the environment is portable
- Archive the tagged release on Zenodo, get a DOI, cite it in the paper
- `git tag -a v1.0-paper` on the commit that produced the reported numbers

Many venues now run artifact evaluation. This is a day of work for a badge.

---

## U9 — Threats to validity and negative results

Two sections most student papers omit and every reviewer notices.

**Threats to validity** — be specific, not ritual:
- Single-host experiments; replica scaling bounded by physical cores
- Bitbrains resampled 5 min → 5 s by interpolation; the high-frequency structure
  is synthetic
- Simulator results where applicable, and how simulator and testbed results
  diverge
- Service-rate μ assumed constant; real services have load-dependent μ
- Tuned on the same workload family evaluated on

**Negative results** — include at least one. Candidates from your own ablations:
disabling the asymmetric loss *improves* MAE; the adaptive margin probably does
little on stationary traces; the predictive advantage vanishes for slow ramps. A
paper reporting only wins reads as under-examined.

---

## Scope reality check

If you want this submitted in a few months, do **U1, U2, U6, U8, U9** — that is
achievable and produces a solid empirical systems paper.

**U3** is what makes it technical. If you have the control-theory background and
the time, it is the highest-leverage item in this document by a wide margin.

**U5** is the strongest extension but is a project of its own.

Do not attempt all nine. A narrow paper with solid evidence beats a broad one
with thin evidence, every time.
