/* The paper. Numbers are read from results/*.csv at build time. */
const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, Footer,
  PageNumber, LevelFormat, convertInchesToTwip, Math: M, MathRun,
  MathFraction, MathSum, MathRadical,
} = d;
const B = require('./build_paper.js');
const { P, RUNS, H1, H2, H3, CAP, BUL, EQ, sub, sup, img, mkTable,
        E1, E2, E3, E4, E5, E6, E7, MAN, by, f, F } = B;

const ROOT = path.resolve(__dirname, '..');

// ---------------------------------------------------------------- data
const po = by(E1, 'method', 'pact-oracle');
const pp = by(E1, 'method', 'pact-persist');
const ro = by(E1, 'method', 'rounded-oracle');
const rp = by(E1, 'method', 'rounded-persist');
const re = by(E1, 'method', 'reactive');
const st = by(E1, 'method', 'static');

const sig = (m, k) => E2.find(r => r.method === m && r.metric === k);
const fr = (m, u) => E6.find(r => r.method === m && Math.abs(r.u_target - u) < 1e-9);

// horizon cliff: for each tau, the smallest H meeting the Eq.19 bound and the
// violation on either side of it
const TRUTHY = v => v === true || v === 'True' || v === 'true';
const taus = [...new Set(E5.map(r => r.tau_c_s))].sort((a, b) => a - b);
const cliff = taus.map(tau => {
  const rows = E5.filter(r => r.tau_c_s === tau).sort((a, b) => a.H - b.H);
  const below = rows.filter(r => !TRUTHY(r.covers_coldstart));
  const above = rows.filter(r => TRUTHY(r.covers_coldstart));
  return {
    tau,
    belowMean: below.length ? below.reduce((s, r) => s + r.pact_sla, 0) / below.length : NaN,
    aboveMean: above.length ? above.reduce((s, r) => s + r.pact_sla, 0) / above.length : NaN,
    firstH: above.length ? above[0].H : NaN,
    bestGain: Math.max(...rows.map(r => r.improvement_pp)),
    worstGain: Math.min(...rows.map(r => r.improvement_pp)),
  };
});
const cliffDrop = cliff.filter(c => Number.isFinite(c.belowMean) && Number.isFinite(c.aboveMean));
const meanBelow = cliffDrop.reduce((s, c) => s + c.belowMean, 0) / cliffDrop.length;
const meanAbove = cliffDrop.reduce((s, c) => s + c.aboveMean, 0) / cliffDrop.length;
const worstNeg = Math.min(...E5.map(r => r.improvement_pp));
const bestPos = Math.max(...E5.map(r => r.improvement_pp));

const e3by = n => {
  const rows = E3.filter(r => r.n_max === n);
  const m = k => rows.reduce((s, r) => s + r[k], 0) / rows.length;
  return { n, dis: m('disagree_pct'), ea: m('exact_actions'), ra: m('rounded_actions'),
           dsla: m('delta_sla'), dover: m('delta_over'), solve: m('exact_solve_ms') };
};
const E3R = [2, 4, 8, 16, 32].map(e3by);
const solveMax = Math.max(...E3R.map(r => r.solve));

const abl = v => E4.find(r => r.variant === v);
const TRUE = v => v === true || v === 'True' || v === 'true';
const noise0 = E7.find(r => r.forecast_noise_sigma === 0 && TRUE(r.safety_guard));
const noiseHi = E7.find(r => r.forecast_noise_sigma === 0.4 && TRUE(r.safety_guard));
const reRef = E7.find(r => r.forecast_noise_sigma === -1);

// ---------------------------------------------------------------- front
const title = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 130 },
    children: [new TextRun({
      text: 'Integer-Exact Predictive Control for Container Auto-Scaling: '
          + 'Measuring the Cost of Relaxation and the Horizon–Dead-Time Law',
      size: 32, bold: true, font: F })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 50 },
    children: [new TextRun({ text: 'Atharva Bandekar', size: 21, font: F })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 260 },
    children: [new TextRun({
      text: 'School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India',
      size: 18, italics: true, font: F, color: '555555' })],
  }),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 230 },
    children: [
      new TextRun({ text: 'Abstract—', size: 18, bold: true, font: F }),
      new TextRun({ size: 18, font: F, text:
        'Container auto-scalers decide how many replicas to run, and that decision is an integer. '
      + 'Published predictive controllers nevertheless solve a relaxed, real-valued problem and round the result, '
      + 'and they rarely state how the forecast horizon should be chosen relative to the time a replica takes to become ready. '
      + 'This paper examines both choices experimentally on a discrete-event pool simulator driven by five synthetic workload families. '
      + 'Because the learned forecaster could not be executed in the available environment, every controller is driven by two bracketing '
      + 'predictors — perfect foresight and persistence — so that controller effects are separated from predictor quality. '
      + 'Two findings follow. First, the forecast horizon must cover the actuation dead time or prediction is actively harmful: '
      + `averaged over the ${cliffDrop.length} cold-start settings for which both sides of the boundary were sampled, violations fall from ${f(meanBelow)} % to ${f(meanAbove)} % as the horizon crosses `
      + `HΔt = τc, and below that threshold the predictive controller is up to ${f(Math.abs(worstNeg))} percentage points worse than a reactive baseline, `
      + `while above it it is up to ${f(bestPos)} points better. Second, relaxing the integrality constraint is not free: the relaxed-and-rounded `
      + `controller disagrees with the integer-exact solution on ${f(E3R[0].dis, 1)}–${f(E3R[4].dis, 1)} % of control ticks and, at an aggressive `
      + `utilisation target of 0.85, reaches ${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations against ${f(fr('pact-oracle', 0.85).sla_mean, 2)} % for the exact controller. `
      + `Exact solution costs under ${f(solveMax, 2)} ms per tick. All reported values are measured; no result in this paper is estimated or illustrative.` }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 240, line: 230 },
    children: [
      new TextRun({ text: 'Index Terms—', size: 18, bold: true, italics: true, font: F }),
      new TextRun({ size: 18, font: F, text:
        'Container auto-scaling, model predictive control, integer programming, actuation dead time, '
      + 'cold start, discrete-event simulation, Docker, resource management.' }),
    ],
  }),
];

// ---------------------------------------------------------------- body
const body = [];

// ===== I. Introduction
body.push(H1('I. Introduction'));
body.push(H2('A. Background'));
body.push(P(
  'A container pool serves traffic with some number of identical replicas, and an auto-scaler adjusts that number as '
+ 'demand moves. The decision is discrete: a pool runs three replicas or four, never 3.4. It is also delayed, because '
+ 'a replica that has been requested is not yet useful. Image pull, runtime initialisation, cache warming, readiness '
+ 'probing and load-balancer registration together give a cold-start interval that is commonly tens of seconds.'));
body.push(P(
  'Threshold auto-scalers, of which the Kubernetes Horizontal Pod Autoscaler is the canonical example, act only after '
+ 'utilisation has crossed a configured target. The corrective action therefore begins after the system has already '
+ 'entered the loaded regime and completes one cold-start interval later still. Predictive auto-scaling removes the '
+ 'first delay by forecasting demand, but it does not by itself remove the second, and whether it removes the problem '
+ 'depends on a relationship between the forecast horizon and the cold-start interval that is seldom made explicit.'));

body.push(H2('B. Research Gap'));
body.push(P(
  'Two gaps motivate this work, and both concern the controller rather than the predictor.'));
body.push(P(
  'The first is integrality. Published predictive auto-scalers that use constrained optimisation formulate the problem '
+ 'over real-valued decision variables and round the solution before applying it. The rounding step is reported as an '
+ 'implementation detail and its cost is not measured. Independent evidence suggests the cost is not negligible: in a '
+ 'recent benchmark of reinforcement-learning controllers, discrete-action algorithms outperformed continuous-action '
+ 'ones, an outcome the authors attribute to the mismatch introduced when continuous outputs are bucketed into integer '
+ 'replica changes. If that mismatch matters for a learned policy, it should also matter for a relaxed optimiser.'));
body.push(P(
  'The second is horizon selection. A predictive controller that looks one step ahead cannot compensate for an actuation '
+ 'delay of nine steps; its decisions arrive exactly as late as a reactive controller’s, with additional forecast error '
+ 'on top. The condition under which prediction becomes useful is simple to state and, as far as the present survey '
+ 'found, is not measured anywhere.'));

body.push(H2('C. Contributions'));
body.push(P('This paper makes three contributions, each supported by measurements reported in Section V.'));
body.push(BUL(
  'A measurement of the horizon–dead-time law. Sweeping the cold-start interval and the forecast horizon independently '
+ 'shows a sharp transition in service-level attainment that coincides with the condition HΔt ≥ τc. Below the transition '
+ 'the predictive controller is worse than a reactive baseline.'));
body.push(BUL(
  'A measurement of the cost of relaxation. An integer-exact controller is compared against an otherwise identical '
+ 'relaxed-and-rounded controller across five pool sizes, isolating the effect of the integrality constraint alone.'));
body.push(BUL(
  'An evaluation protocol that separates controller effects from predictor quality by bracketing every controller between '
+ 'a perfect-foresight predictor and a persistence predictor, so that conclusions holding under both brackets are '
+ 'attributable to the control law.'));

body.push(H2('D. Organisation'));
body.push(P(
  'Section II reviews existing auto-scaling systems and groups the recent literature by the mechanism each study adopts. '
+ 'Section III develops the controller. Section IV describes the experimental setup, the simulated plant and the workloads. '
+ 'Section V reports results. Section VI states the limitations, several of which are substantial. Section VII concludes.'));

// ===== II. Literature
body.push(H1('II. Existing Systems and Related Work'));
body.push(H2('A. Threshold-Based Auto-Scaling in Production Orchestrators'));
body.push(P(
  'The auto-scalers actually deployed at scale are reactive and rule-based. The Kubernetes Horizontal Pod Autoscaler '
+ 'samples an aggregate utilisation metric on a fixed interval and sets the replica count to the current count scaled by '
+ 'the ratio of observed to target utilisation, subject to a tolerance band and a scale-down stabilisation window. '
+ 'Docker Swarm and the managed service schedulers offered by the major cloud providers follow the same pattern. '
+ 'Van Do et al. (2025) characterise the HPA control law analytically and show that only a finite set of tolerance '
+ 'intervals produces distinct scaling decisions, which both simplifies configuration and bounds what the mechanism can '
+ 'achieve. The common limitation is structural rather than a matter of tuning: a threshold cannot fire before the '
+ 'threshold is crossed, so every correction begins late and completes one cold-start interval later.'));

body.push(H2('B. Forecast-Driven Scaling'));
body.push(P(
  'The dominant response has been to place a time-series model ahead of the scaling rule. Reported architectures span '
+ 'statistical models, recurrent networks, and attention-based sequence models; Jeong and Jeong (2025) and a systematic '
+ 'review of sixty-four primary studies both find recurrent networks, and long short-term memory variants in particular, '
+ 'to dominate the forecasting stage. Within this group the differences are largely architectural, and the evaluation is '
+ 'almost always reported as forecasting error. Fog et al. (2025) compare neural and statistical forecasters under a '
+ 'common protocol and find the advantage of neural models to be workload-dependent rather than universal, with classical '
+ 'models competitive on smooth or strongly seasonal traffic at a fraction of the computational cost. The recurring '
+ 'limitation of this group is that a forecast is not a decision: the mapping from a predicted utilisation series to an '
+ 'integer replica count is where the operational outcome is determined, and it is frequently left unspecified.'));

body.push(H2('C. Optimisation-Based and Control-Theoretic Scaling'));
body.push(P(
  'A smaller body of work treats scaling as a control problem. Baresi et al. (2018) combine vertical and horizontal '
+ 'scaling through model predictive control. Quattrocchi et al. (2024) estimate system state with a queueing network '
+ 'and compute allocations at a one-second control interval. Zhang et al. (2024), in the OptScaler framework deployed at '
+ 'Alipay, couple a proactive forecaster and a reactive estimator into a single receding-horizon optimisation with a '
+ 'chance constraint, applying only the first step of each solution. Cai et al. (2025) pair a deep forecaster with a '
+ 'feedback control law so that the learned term supplies feedforward action while the feedback term corrects residual '
+ 'error. These are the closest antecedents of the present work, and two properties recur across them. Their decision '
+ 'variables are real-valued and the solution is rounded before use; and none reports a stability analysis, an '
+ 'oscillation metric, or a penalty on changing the decision between intervals.'));

body.push(H2('D. Actuation Delay as an Explicit Quantity'));
body.push(P(
  'Treating start-up latency as a first-class quantity is recent. Work on predictive scaling for large language model '
+ 'serving, where loading model weights can take minutes, decomposes predictive auto-scaling into demand tracking, '
+ 'startup-delay lookahead, an uncertainty margin and plant-state observation, and reports that lookahead alone accounts '
+ 'for the largest share of the improvement. That study operates in a regime two orders of magnitude slower than '
+ 'container start-up and uses an exponentially weighted predictor with a confidence-bound margin. Its central '
+ 'observation — that lookahead dominates — motivates the question this paper measures directly: how much lookahead, '
+ 'relative to what delay.'));

body.push(H2('E. Gap Addressed Here'));
body.push(P(
  'Across these groups, the integrality of the scaling decision is treated as a rounding detail and the relationship '
+ 'between horizon and actuation delay is treated as a configuration choice. Neither has been measured. This paper '
+ 'measures both, using a protocol that holds the predictor fixed at two extremes so that the controller is the only '
+ 'variable under test.'));

// ===== III. Methodology
body.push(H1('III. Controller Design'));
body.push(P(
  'The controller observes the pool once per interval Δt, converts a demand forecast into an integer replica requirement, '
+ 'and solves a constrained receding-horizon problem whose decision variable is the change in replica count. Fig. 1 shows '
+ 'the implemented pipeline; each block is annotated with the module that realises it, and blocks that exist in the '
+ 'repository but were not exercised in this evaluation are drawn with a dashed outline.'));

const SPLIT = body.length;   // wide architecture figure goes here

body.push(H2('A. Observation and Capacity Mapping'));
body.push(P('At each tick the collector assembles the observation vector of (1), where u is mean CPU utilisation across '
+ 'the pool, r is the resident memory fraction, λ the request arrival rate, ℓ the 95th-percentile response time, and N '
+ 'the current replica count.'));
body.push(EQ([new MathRun('m(t) = [ u(t), r(t), λ(t), ℓ(t), N(t) ]'), sup('', 'T')], 1));
body.push(P('A forecast of utilisation is converted to a replica requirement through the utilisation law (2), in which '
+ 'u* is the target operating point, μ the per-replica service rate, and γ a safety margin. The ceiling is used because '
+ 'a fractional shortfall is a real shortfall.'));
body.push(EQ([
  new MathRun('Ñ(t+h) = ⌈ (1 + γ) · '),
  new MathFraction({ numerator: [new MathRun('λ̂(t+h)')],
                     denominator: [new MathRun('μ u*')] }),
  new MathRun(' ⌉'),
], 2));
body.push(P('Memory imposes an independent bound, so the binding requirement is the larger of the two, as in (3).'));
body.push(EQ([
  new MathRun('Ñ(t+h) ← max{ '), sub('Ñ', 'CPU'),
  new MathRun('(t+h), ⌈ r̂(t+h) N(t) / r* ⌉ }'),
], 3));
body.push(P('Expected tail latency at a candidate replica count is estimated by the heavy-traffic approximation (4), '
+ 'where s̄ is the mean service time and cₐ² and cₛ² are the squared coefficients of variation of the arrival and '
+ 'service processes. It is used as a feasibility test: candidates whose predicted latency exceeds the objective are '
+ 'rejected before the optimisation runs.'));
body.push(EQ([
  new MathRun('ℓ̂ = s̄ + '),
  new MathFraction({ numerator: [sub('c', 'a'), new MathRun('² + '), sub('c', 's'), new MathRun('²')],
                     denominator: [new MathRun('2')] }),
  new MathFraction({ numerator: [new MathRun('ρ')],
                     denominator: [new MathRun('1 − ρ')] }),
  new MathRun(' s̄, ρ = λ̂ / (Nμ)'),
], 4));

body.push(H2('B. Plant Model and Dead Time'));
body.push(P('The pool evolves according to (5): the replica count changes by the control input applied τ ticks earlier, '
+ 'where τ is the cold-start interval expressed in control ticks. This delay term is what forces the controller to act '
+ 'before demand arrives rather than when it arrives.'));
body.push(EQ([
  new MathRun('N(t+1) = N(t) + u(t − τ), τ = ⌈ '), sub('τ', 'c'),
  new MathRun(' / Δt ⌉, u ∈ ℤ'),
], 5));
body.push(P('For the forecast to be actionable it must reach at least τ ticks into the future, which gives the '
+ 'feasibility condition (6). Section V measures the behaviour of the closed loop on both sides of this inequality; '
+ 'it is the paper’s central design rule and the subject of its first result.'));
body.push(EQ([
  new MathRun('H Δt ≥ '), sub('τ', 'c'), new MathRun(' + '), sub('Δt', 'ctrl'),
], 6));

body.push(H2('C. Receding-Horizon Objective'));
body.push(P('The applied action minimises (7) over the horizon. The three terms penalise, in order, deviation from the '
+ 'requirement, the magnitude of the action, and change in the action between consecutive steps. The third term is what '
+ 'suppresses the alternating scale-up and scale-down behaviour that threshold auto-scalers exhibit under bursty load.'));
body.push(EQ([
  new MathRun('J(u) = '),
  new MathSum({
    children: [
      new MathRun('[ '), sub('q', 'h'),
      new MathRun('( N(t+h) − Ñ(t+h) )² + r u(t+h−1)² + s ( u(t+h−1) − u(t+h−2) )² ]'),
    ],
    subScript: [new MathRun('h=1')], superScript: [new MathRun('H')],
  }),
], 7));
body.push(P('The tracking weight is asymmetric, as in (8): being short of the requirement is penalised more heavily than '
+ 'being over it, because a dropped request costs more than an idle replica-second.'));
body.push(EQ([
  sub('q', 'h'), new MathRun(' = '), sup('q', '+'),
  new MathRun(' if N(t+h) < Ñ(t+h), else '), sup('q', '−'),
  new MathRun(', '), sup('q', '+'), new MathRun(' ≫ '), sup('q', '−'),
], 8));
body.push(P('The optimisation is subject to (9): the pool is bounded, the per-tick change is rate limited, and the '
+ 'decision is an integer. The last of these is the constraint that related work relaxes.'));
body.push(EQ([
  sub('N', 'min'), new MathRun(' ≤ N(t+h) ≤ '), sub('N', 'max'),
  new MathRun(', | u | ≤ '), sub('Δ', 'max'), new MathRun(', u ∈ ℤ'),
], 9));
body.push(P('Because the horizon is short and the action set is a bounded integer interval, the exact problem is solved '
+ 'by enumeration over the admissible first actions rather than by relaxation. Section V reports the cost of this choice '
+ `in both directions: the solve time, which stays below ${f(solveMax, 2)} ms per tick, and the decisions it changes.`));
body.push(P('Only the first element of the optimal sequence is applied and the problem is re-solved at the next tick. '
+ 'A deadband and a cooldown then gate the result, as in (10); the gates can suppress an action but never create one, '
+ 'and they are applied after the feasibility projections so that they cannot be defeated by an unusual forecast.'));
body.push(EQ([
  sub('u', 'applied'), new MathRun('(t) = u*(t) if | u*(t) | ≥ θ and t − '),
  sub('t', 'up'), new MathRun(' ≥ '), sub('T', 'cool'), new MathRun(', else 0'),
], 10));

body.push(H2('D. Margin Adaptation and Safety Ceiling'));
body.push(P('A drift monitor compares the replica count the observed load actually required against the count predicted '
+ 'τ ticks earlier and adapts the margin by the clipped integral update (11). A separate ceiling bypasses the optimiser '
+ 'and every gate when observed utilisation exceeds an emergency threshold, so that an unprecedented burst is not held '
+ 'back by the deadband. Section V reports that neither mechanism was active under the conditions tested.'));
body.push(EQ([
  new MathRun('γ(t+1) = clip( γ(t) + '), sub('κ', 'γ'),
  new MathRun(' ē(t), '), sub('γ', 'min'), new MathRun(', '), sub('γ', 'max'), new MathRun(' )'),
], 11));

// ===== IV. Setup
body.push(H1('IV. Experimental Setup'));
body.push(H2('A. Plant and Environment'));
body.push(P(
  'Experiments run against the discrete-event pool simulator in the project repository. The simulator maintains a set of '
+ 'ready replicas and a set of pending replicas; a replica requested at tick t becomes ready at tick t + τ and contributes '
+ 'no capacity before then, so the cold-start interval is modelled explicitly rather than absorbed into a margin. '
+ 'Tail latency is computed from the heavy-traffic approximation of (4). When offered load exceeds pool capacity the '
+ `approximation diverges; such ticks are censored at ${'5000'} ms and the censoring rate is reported with every mean.`));
body.push(RUNS([
  { t: 'Environment. ', b: true },
  { t: `All runs executed on ${MAN.platform}, Python ${MAN.python}, NumPy ${MAN.numpy}, `
     + `at repository commit ${String(MAN.git_sha).slice(0, 10)}. Each run is ${MAN.n_ticks_per_run} control ticks at `
     + `Δt = 5 s, giving a simulated duration of ${(MAN.n_ticks_per_run * 5 / 60).toFixed(0)} minutes. `
     + `Results are reported over ${MAN.n_seeds} seeds unless stated otherwise.` },
]));

body.push(H2('B. Datasets'));
body.push(P(
  'The arrival traces are synthetic and generated by the harness in the repository; they are not drawn from any public '
+ 'dataset. This is a limitation and is stated as such in Section VI. Five families are used, chosen to span distinct '
+ 'demand shapes rather than to sample one shape repeatedly. Each is parameterised by a seed that perturbs it with '
+ 'additive Gaussian noise, and the same seed produces the same trace for every method, so comparisons are paired.'));
body.push(mkTable(
  ['Trace family', 'Shape', 'Provenance', 'Ticks per run', 'Seeds'],
  [
    ['Diurnal', 'Sinusoidal base, period 300 ticks', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Burst', 'Two Gaussian bursts on a flat base', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Ramp', 'Linear rise, plateau, linear fall', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Step', 'Single discontinuity at mid-run', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Sawtooth', 'Repeating linear rise, period 90 ticks', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
  ],
  [1700, 2600, 1900, 1100, 800], { left: true }));
body.push(CAP('Table I. Workload families. All traces are generated by experiments/harness.py in the project '
+ 'repository; none is a public trace. Every method sees the identical trace for a given family and seed.'));

body.push(H2('C. Forecaster Bracketing'));
body.push(P(
  'The learned forecaster could not be executed: the environment has no package-index access and PyTorch is unavailable, '
+ 'as are statsmodels and the deep baselines that depend on them. Rather than substitute a different model and present it '
+ 'as the proposed one, every controller is driven by two bracketing predictors. The oracle predictor has perfect '
+ 'foresight of the arrival series and represents the limit of what forecasting can contribute. The persistence predictor '
+ 'repeats the most recent observation across the horizon and represents the absence of a model. Any real forecaster lies '
+ 'between them. Conclusions that hold under both brackets are properties of the control law rather than of the predictor, '
+ 'and only such conclusions are claimed.'));

body.push(H2('D. Methods Compared'));
body.push(P(
  'Five decision mechanisms are compared. All receive identical telemetry, run at the same control interval, actuate '
+ 'through the same path, and are subject to the same pool bounds and rate limit; only the decision rule differs.'));
body.push(BUL('Integer-exact MPC, the controller of Section III, solving (7)–(9) exactly over the admissible integer actions.'));
body.push(BUL('Relaxed-and-rounded MPC, identical in objective and constraints except that the integrality constraint of (9) '
+ 'is dropped and the first element of the solution is rounded before application. This reproduces the choice made by '
+ 'the optimisation-based systems reviewed in Section II-C.'));
body.push(BUL('A reactive threshold controller implementing the HPA control law with its tolerance band and scale-down '
+ 'stabilisation window.'));
body.push(BUL('A static configuration fixed at the level required to serve the peak of the trace.'));

body.push(H2('E. Metrics and Statistics'));
body.push(P(
  'Service-level violation rate is the fraction of ticks whose tail latency exceeds the objective. Over-provisioning '
+ 'ratio is the mean positive excess of the replica count over the requirement. Actuation churn is the number of ticks '
+ 'on which the replica count changed. Cost is replica-seconds per thousand requests served. Comparisons against the '
+ 'proposed controller use the Wilcoxon signed-rank test on paired runs, with Holm correction across the family of '
+ 'comparisons and Cliff’s delta as an effect size; confidence intervals are bias-corrected and accelerated bootstrap '
+ 'intervals over 4000 resamples. Significance is never claimed without the corrected p-value.'));

// ===== V. Results
body.push(H1('V. Results and Discussion'));
body.push(P(
  `All values in this section are computed from the run logs written to the repository’s results directory. Tables II `
+ `through VI and Figs. 2 through 7 are generated directly from those files.`));

body.push(H2('A. The Horizon–Dead-Time Law'));
body.push(P(
  'The first experiment sweeps the cold-start interval τc over six values and the forecast horizon H over ten, and '
+ 'measures the resulting violation rate for the proposed controller and for the reactive baseline. Fig. 2 shows the '
+ 'outcome as a heat map with the boundary HΔt = τc from (6) drawn on it, and Fig. 3 shows the same data as violation '
+ 'against lookahead for each cold-start setting.'));
body.push(P(
  'The transition is sharp and it coincides with the predicted boundary. For every cold-start setting tested, violation '
+ 'rate is flat and elevated while the horizon is too short, then falls abruptly at the first horizon satisfying '
+ `HΔt ≥ τc, and is flat and low thereafter. Averaged over the six settings, the rate falls from ${f(meanBelow)} % below `
+ `the boundary to ${f(meanAbove)} % above it. The effect is monotone in τc: at τc = ${cliff[0].tau} s the controller is already `
+ `near its floor at the shortest horizon tested, whereas at τc = ${cliff[cliff.length - 1].tau} s it remains at `
+ `${f(cliff[cliff.length - 1].belowMean)} % until the horizon reaches ${cliff[cliff.length - 1].firstH} ticks.`));
body.push(P(
  'The comparison against the reactive baseline is the operationally important part, and it is not flattering to naive '
+ 'predictive scaling. When the horizon does not cover the dead time the predictive controller is not merely no better '
+ `than reacting; it is worse, by up to ${f(Math.abs(worstNeg))} percentage points. Once the horizon covers the dead time it is better, by up `
+ `to ${f(bestPos)} points. A predictive auto-scaler configured with an under-sized horizon therefore pays the cost of forecasting `
+ 'and receives none of the benefit, and the sign of its advantage is determined by a condition that is cheap to check '
+ 'at configuration time. This is the clearest result in the paper and it is the one with the most direct practical use.'));
body.push(img('fig_horizon_law.png', 3.35));
body.push(CAP('Fig. 2. Violation rate over the cold-start × horizon grid for the proposed controller. The solid line is '
+ 'the boundary HΔt = τc from (6). Synthetic traces; mean of three families × three seeds per cell.'));
body.push(img('fig_horizon_cliff.png', 3.3));
body.push(CAP('Fig. 3. The same data as violation against lookahead. Dotted verticals mark τc for each curve. '
+ 'Synthetic traces.'));

body.push(H2('B. The Cost of Relaxing Integrality'));
body.push(P(
  'The second experiment compares the integer-exact controller against the relaxed-and-rounded controller across five '
+ 'pool bounds, holding everything else fixed. Table II reports the result and Fig. 4 shows it graphically.'));
body.push(mkTable(
  ['Nₘₐₓ', 'Ticks where\ndecisions differ (%)', 'Churn, exact', 'Churn, rounded', 'Δ violation (pp)', 'Mean solve (ms)'],
  E3R.map(r => [String(r.n), f(r.dis, 1), f(r.ea, 1), f(r.ra, 1), f(r.dsla, 2), f(r.solve, 3)]),
  [900, 2100, 1500, 1600, 1500, 1500]));
body.push(CAP('Table II. Integer-exact against relaxed-and-rounded control, by pool bound. Synthetic traces, five '
+ 'families × five seeds per row. A positive Δ violation means the rounded controller violated more often.'));
body.push(P(
  `The two controllers disagree on a substantial and increasing fraction of ticks, from ${f(E3R[0].dis, 1)} % at the smallest `
+ `pool to ${f(E3R[4].dis, 1)} % at the largest. The disagreement does not translate into a large difference in violation rate at `
+ `this operating point — the mean difference ranges from ${f(Math.min(...E3R.map(r => r.dsla)), 2)} to `
+ `${f(Math.max(...E3R.map(r => r.dsla)), 2)} percentage points — but it translates strongly into churn. At Nₘₐₓ = 16 the exact `
+ `controller issues ${f(E3R[3].ea, 0)} scaling actions per run against ${f(E3R[3].ra, 0)} for the rounded controller, and at `
+ `Nₘₐₓ = 32 the figures are ${f(E3R[4].ea, 0)} against ${f(E3R[4].ra, 0)}. Rounding converts a fractional preference into a `
+ 'full replica change, and because the fractional part fluctuates from tick to tick the rounded controller oscillates '
+ 'where the exact controller holds.'));
body.push(P(
  `The cost of solving exactly is small. The enumeration completes in under ${f(solveMax, 2)} ms per tick at the largest pool `
+ 'tested, which is three orders of magnitude inside the 5 s control interval. On this evidence the integrality '
+ 'constraint should not be relaxed for performance reasons at these problem sizes, because there is no performance '
+ 'reason to relax it.'));
body.push(img('fig_rounding_gap.png', 6.6));
body.push(CAP('Fig. 4. Disagreement between the exact and rounded controllers, and the resulting actuation churn. '
+ 'Synthetic traces.'));

body.push(H2('C. Operating-Point Sensitivity'));
body.push(P(
  'Comparing controllers at a single target utilisation is misleading, because each sits at a different point on the '
+ 'trade-off between service level and capacity. The third experiment therefore sweeps the target from 0.50 to 0.85 and '
+ 'traces the attainable frontier for each method. Table III reports the extremes and Fig. 5 shows the full sweep.'));
body.push(mkTable(
  ['Target u*', 'Exact MPC\nviol. (%)', 'Exact MPC\nover (%)', 'Rounded\nviol. (%)', 'Rounded\nover (%)', 'Reactive\nviol. (%)', 'Reactive\nover (%)'],
  [0.50, 0.65, 0.75, 0.85].map(u => [
    f(u, 2),
    f(fr('pact-oracle', u).sla_mean, 2), f(fr('pact-oracle', u).over_mean, 1),
    f(fr('rounded-oracle', u).sla_mean, 2), f(fr('rounded-oracle', u).over_mean, 1),
    f(fr('reactive', u).sla_mean, 2), f(fr('reactive', u).over_mean, 1),
  ]),
  [1100, 1400, 1400, 1300, 1300, 1300, 1300]));
body.push(CAP('Table III. Service level and capacity cost against target utilisation. Synthetic traces, five families '
+ '× three seeds per cell, oracle predictor.'));
body.push(P(
  'The two controllers degrade very differently as the target is pushed up. The rounded controller is efficient and '
+ `adequate at conservative targets — at u* = 0.50 it violates on ${f(fr('rounded-oracle', 0.50).sla_mean, 2)} % of ticks while carrying only `
+ `${f(fr('rounded-oracle', 0.50).over_mean, 1)} % excess capacity — but it collapses as the target rises, reaching `
+ `${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations at u* = 0.85. The exact controller rises far more slowly over the same range, from `
+ `${f(fr('pact-oracle', 0.50).sla_mean, 2)} % to ${f(fr('pact-oracle', 0.85).sla_mean, 2)} %. The mechanism is visible in Table II: when the pool is run close to `
+ 'its limit, a rounding error of one replica is a large fraction of the available headroom, and the rounded controller '
+ 'spends an increasing share of its time on the wrong side of the requirement.'));
body.push(P(
  'The honest reading of this experiment is that neither controller dominates. The exact controller buys service level '
+ `with capacity: at u* = 0.65 it carries ${f(fr('pact-oracle', 0.65).over_mean, 1)} % excess against the rounded controller’s `
+ `${f(fr('rounded-oracle', 0.65).over_mean, 1)} %, for a violation rate of ${f(fr('pact-oracle', 0.65).sla_mean, 2)} % against `
+ `${f(fr('rounded-oracle', 0.65).sla_mean, 2)} %. An operator who can tolerate the violation rate of the rounded controller should prefer it on cost. `
+ 'The case for exact solution is that it remains usable at operating points where the rounded controller does not, and '
+ 'that it costs nothing to obtain.'));
body.push(img('fig_frontier.png', 3.35));
body.push(CAP('Fig. 5. Attainable frontier as the target utilisation is swept from 0.50 to 0.85. Synthetic traces, '
+ 'oracle predictor. Points are labelled with the target that produced them.'));

body.push(H2('D. Aggregate Comparison and Significance'));
body.push(P(
  `Table IV reports the main comparison over ${MAN.workloads.length} workload families × ${MAN.n_seeds} seeds, with bootstrap confidence `
+ 'intervals, under both forecaster brackets. Table V reports the paired significance tests against the proposed '
+ 'controller with the oracle predictor.'));
body.push(mkTable(
  ['Method', 'Predictor', 'Violation (%)\nmean [95% CI]', 'Over-prov. (%)\nmean [95% CI]', 'Churn', 'Util. (%)', 'Cost /1k req'],
  [
    ['Integer-exact MPC', 'Oracle', `${f(po.sla_mean)} [${f(po.sla_ci_lo)}, ${f(po.sla_ci_hi)}]`, `${f(po.over_mean, 1)} [${f(po.over_ci_lo, 1)}, ${f(po.over_ci_hi, 1)}]`, f(po.act_mean, 1), f(po.util_mean, 1), f(po.cost_mean, 1)],
    ['Integer-exact MPC', 'Persistence', `${f(pp.sla_mean)} [${f(pp.sla_ci_lo)}, ${f(pp.sla_ci_hi)}]`, `${f(pp.over_mean, 1)} [${f(pp.over_ci_lo, 1)}, ${f(pp.over_ci_hi, 1)}]`, f(pp.act_mean, 1), f(pp.util_mean, 1), f(pp.cost_mean, 1)],
    ['Relaxed + rounded', 'Oracle', `${f(ro.sla_mean)} [${f(ro.sla_ci_lo)}, ${f(ro.sla_ci_hi)}]`, `${f(ro.over_mean, 1)} [${f(ro.over_ci_lo, 1)}, ${f(ro.over_ci_hi, 1)}]`, f(ro.act_mean, 1), f(ro.util_mean, 1), f(ro.cost_mean, 1)],
    ['Relaxed + rounded', 'Persistence', `${f(rp.sla_mean)} [${f(rp.sla_ci_lo)}, ${f(rp.sla_ci_hi)}]`, `${f(rp.over_mean, 1)} [${f(rp.over_ci_lo, 1)}, ${f(rp.over_ci_hi, 1)}]`, f(rp.act_mean, 1), f(rp.util_mean, 1), f(rp.cost_mean, 1)],
    ['Reactive threshold', '—', `${f(re.sla_mean)} [${f(re.sla_ci_lo)}, ${f(re.sla_ci_hi)}]`, `${f(re.over_mean, 1)} [${f(re.over_ci_lo, 1)}, ${f(re.over_ci_hi, 1)}]`, f(re.act_mean, 1), f(re.util_mean, 1), f(re.cost_mean, 1)],
    ['Static over-provisioned', '—', `${f(st.sla_mean)} [${f(st.sla_ci_lo)}, ${f(st.sla_ci_hi)}]`, `${f(st.over_mean, 1)} [${f(st.over_ci_lo, 1)}, ${f(st.over_ci_hi, 1)}]`, f(st.act_mean, 1), f(st.util_mean, 1), f(st.cost_mean, 1)],
  ],
  [2000, 1300, 2300, 2300, 900, 1000, 1200], { left: false, size: 14 }));
body.push(CAP(`Table IV. Main comparison. Synthetic traces, ${MAN.workloads.length} families × ${MAN.n_seeds} seeds = ${po.n_runs} runs per row. `
+ 'Intervals are 95 % bias-corrected bootstrap over 4000 resamples.'));

const sigRows = [
  ['Relaxed + rounded (oracle)', 'Violation', sig('rounded-oracle', 'sla')],
  ['Relaxed + rounded (oracle)', 'Over-provisioning', sig('rounded-oracle', 'over')],
  ['Relaxed + rounded (oracle)', 'Churn', sig('rounded-oracle', 'act')],
  ['Reactive threshold', 'Violation', sig('reactive', 'sla')],
  ['Reactive threshold', 'Over-provisioning', sig('reactive', 'over')],
  ['Reactive threshold', 'Churn', sig('reactive', 'act')],
  ['Static over-provisioned', 'Violation', sig('static', 'sla')],
  ['Static over-provisioned', 'Over-provisioning', sig('static', 'over')],
].filter(r => r[2]);
body.push(mkTable(
  ['Comparison', 'Metric', 'Proposed', 'Other', 'p (Holm)', 'Cliff’s δ', 'Significant'],
  sigRows.map(([name, metric, r]) => [
    name, metric, f(r.ref_mean, 2), f(r.other_mean, 2),
    r.p_holm < 0.001 ? '< 0.001' : f(r.p_holm, 3),
    f(r.cliffs_delta, 2), r['significant_0.05'] === true || r['significant_0.05'] === 'True' ? 'yes' : 'no',
  ]),
  [2600, 1600, 1100, 1000, 1200, 1100, 1200], { left: false, size: 14 }));
body.push(CAP('Table V. Paired Wilcoxon signed-rank tests against the integer-exact controller with the oracle '
+ 'predictor, Holm-corrected across the comparison family. n = 50 paired runs. Synthetic traces.'));
body.push(P(
  'Two aspects of Table V deserve emphasis because they qualify the paper’s claims. First, the difference in violation '
+ `rate between the proposed controller and the rounded controller at this operating point is not significant after `
+ `correction (p = ${f(sig('rounded-oracle', 'sla').p_holm, 3)}), and neither is the difference against the reactive baseline `
+ `(p = ${f(sig('reactive', 'sla').p_holm, 3)}). The service-level advantage reported in Section V-C emerges at aggressive operating `
+ 'points, not at the default one. Second, the differences that are significant are in churn and capacity: against the '
+ `rounded controller the reduction in churn has a Cliff’s delta of ${f(sig('rounded-oracle', 'act').cliffs_delta, 2)}, and against the reactive `
+ `baseline the reduction in over-provisioning has a delta of ${f(sig('reactive', 'over').cliffs_delta, 2)}. The robust claims are about stability `
+ 'and cost, not about service level in the ordinary regime.'));
body.push(P(
  'The forecaster brackets behave as intended and bound the predictor’s contribution. Moving from perfect foresight to '
+ `persistence raises violations from ${f(po.sla_mean)} % to ${f(pp.sla_mean)} % and churn from ${f(po.act_mean, 0)} to ${f(pp.act_mean, 0)} actions, so a realistic `
+ 'forecaster would land inside an interval of roughly a quarter of a percentage point in violation rate. The ordering '
+ 'of methods is unchanged under both brackets, which is the condition under which the comparison is attributed to the '
+ 'control law.'));
body.push(img('fig_trace.png', 6.6));
body.push(CAP('Fig. 6. Replica count and tail latency on one burst trace. The rounded controller overshoots at each '
+ 'burst and the reactive controller holds capacity long after demand falls. Synthetic burst trace, seed 1001, oracle '
+ 'predictor.'));
body.push(img('fig_main_comparison.png', 6.6));
body.push(CAP(`Fig. 7. Main comparison with 95 % bootstrap intervals. Synthetic traces, ${po.n_runs} runs per bar.`));

body.push(H2('E. Ablation'));
body.push(P(
  'Each component of the controller was disabled in turn, with everything else held fixed, over the same five workload '
+ `families and ${MAN.n_seeds} seeds. Table VI reports the outcome and Fig. 8 shows it graphically. The results are more `
+ 'negative than anticipated and are reported as measured.'));
body.push(P(
  'Removing the optimisation layer entirely, so that the controller applies the first element of the demand trajectory '
+ `directly, changes the violation rate by ${f(abl('no_mpc').sla_delta_vs_full, 2)} percentage points and reduces capacity cost by `
+ `${f(Math.abs(abl('no_mpc').over_delta_vs_full), 1)} points. The change in service level is not significant `
+ `(p = ${f(abl('no_mpc').sla_p_raw, 3)} uncorrected). Removing the churn penalty leaves service level essentially unchanged at `
+ `${f(abl('no_churn').sla_delta_vs_full, 2)} points but raises over-provisioning by ${f(abl('no_churn').over_delta_vs_full, 1)} points and churn by `
+ `${f(abl('no_churn').act_delta_vs_full, 0)} actions, which is the clearest single-component effect in the table and is consistent with the `
+ 'rounding result of Section V-B: penalising change is what keeps the pool steady.'));
body.push(P(
  'Two components had no measurable effect under these conditions and this should be stated plainly. Disabling cold-start '
+ `compensation within the controller changed the violation rate by ${f(abl('no_coldstart').sla_delta_vs_full, 2)} points `
+ `(p = ${f(abl('no_coldstart').sla_p_raw, 3)}), and disabling the adaptive margin changed nothing at all: the margin remained at its initial `
+ 'value throughout every run, because the tracking error under both forecaster brackets never accumulated enough to move '
+ 'the clipped integral. The absence of a cold-start effect within the controller is not a contradiction of Section V-A; '
+ 'in that experiment the horizon itself was varied, whereas here the horizon remains long enough to cover the dead time '
+ 'in every variant, so the explicit delay term has little left to correct. The adaptive margin, on this evidence, is '
+ 'unexercised machinery, and a study designed to stress it would need a predictor that degrades over time rather than one '
+ 'that is either perfect or memoryless.'));
body.push(mkTable(
  ['Variant', 'Violation (%)', 'Δ vs. full (pp)', 'Over-prov. (%)', 'Δ vs. full (pp)', 'Churn', 'p (uncorr.)'],
  E4.filter(r => r.variant !== 'full').map(r => [
    ({ no_mpc: 'Without MPC layer', no_churn: 'Without churn penalty',
       no_coldstart: 'Without cold-start comp.', no_margin: 'Without adaptive margin',
       no_safety: 'Without safety ceiling' })[r.variant],
    f(r.sla_mean, 2), f(r.sla_delta_vs_full, 2), f(r.over_mean, 1),
    f(r.over_delta_vs_full, 1), f(r.act_mean, 1),
    r.sla_p_raw === '' ? '—' : f(r.sla_p_raw, 3),
  ]),
  [2300, 1300, 1400, 1300, 1400, 900, 1200], { left: false, size: 14 }));
body.push(CAP(`Table VI. Ablation. Full controller: violation ${f(abl('full').sla_mean, 2)} %, over-provisioning `
+ `${f(abl('full').over_mean, 1)} %, churn ${f(abl('full').act_mean, 1)}. Synthetic traces, ${abl('full').n_runs} runs per row, oracle predictor. `
+ 'No service-level delta reaches p < 0.05.'));
body.push(img('fig_ablation.png', 6.6));
body.push(CAP('Fig. 8. Effect of disabling each component. Synthetic traces.'));

body.push(H2('F. Robustness to Forecast Corruption'));
body.push(P(
  'The final experiment injects multiplicative noise into the series seen by the forecaster while the plant continues to '
+ 'receive the true arrivals, sweeping the corruption magnitude and repeating the sweep with the safety ceiling disabled. '
+ 'Fig. 9 shows the result, which is counter-intuitive and worth stating carefully.'));
body.push(P(
  `Violation rate does not rise with corruption; it falls slightly, from ${f(noise0.sla_mean, 3)} % at zero noise to `
+ `${f(noiseHi.sla_mean, 3)} % at σ = 0.4, while over-provisioning rises from ${f(noise0.over_mean, 1)} % to ${f(noiseHi.over_mean, 1)} %. `
+ 'The explanation is the asymmetric tracking weight of (8). Multiplicative noise causes the forecast to over-predict as '
+ 'often as it under-predicts, but the controller responds to over-prediction by adding capacity and to under-prediction '
+ 'by removing it, and the asymmetric penalty makes the former cheaper. Noise therefore acts as an accidental safety '
+ 'margin, purchased with capacity. This is a property of the cost function rather than a desirable robustness result, '
+ 'and it means the controller is not stress-tested by symmetric noise; a corruption model with sustained negative bias '
+ 'would be a sharper test and is left for future work.'));
body.push(P(
  `The safety ceiling never activated at any corruption level, so the two curves coincide exactly. The reactive baseline, `
+ `measured on the same traces, violates on ${f(reRef.sla_mean, 2)} % of ticks, above every point of the swept range. The ceiling is `
+ 'therefore untested by this experiment rather than shown to be unnecessary.'));
body.push(img('fig_robustness.png', 3.35));
body.push(CAP('Fig. 9. Violation rate against forecast corruption, with and without the safety ceiling. Synthetic '
+ 'traces, three families × four seeds per point.'));

// ===== VI. Limitations
body.push(H1('VI. Limitations and Threats to Validity'));
body.push(P(
  'This evaluation has substantial limitations. They are listed here rather than distributed through the text because '
+ 'several of them bound the claims directly.'));
body.push(RUNS([{ t: 'Simulated plant only. ', b: true }, { t:
  'No result in this paper was obtained on real containers. The plant is the analytical queueing simulator described in '
+ 'Section IV-A, and the cold-start interval is a parameter of that simulator rather than a measurement. Image pull '
+ 'variance, scheduler interference, connection draining and load-balancer registration delay are all absent. The '
+ 'horizon–dead-time law of Section V-A is a property of the model as much as of the controller, and confirming it on a '
+ 'real Docker deployment is the single most important piece of outstanding work.' }]));
body.push(RUNS([{ t: 'Synthetic workloads. ', b: true }, { t:
  'The five trace families are generated, not recorded. Public traces could not be retrieved because the environment has '
+ 'no network access. Real traffic has heavier tails and non-stationarity that these traces do not reproduce.' }]));
body.push(RUNS([{ t: 'The learned forecaster was not executed. ', b: true }, { t:
  'PyTorch is unavailable in the environment and could not be installed, so the temporal convolutional forecaster, and '
+ 'the recurrent and attention-based baselines that would be compared against it, were not run. The bracketing protocol '
+ 'of Section IV-C bounds the effect of this substitution but does not eliminate it: no claim about forecasting accuracy '
+ 'is made anywhere in this paper, and the comparison against statistical and deep forecasting baselines remains '
+ 'outstanding. For the same reason the ARIMA baseline, which requires statsmodels, was not run.' }]));
body.push(RUNS([{ t: 'Two mechanisms are unexercised. ', b: true }, { t:
  'The adaptive margin never moved from its initial value and the safety ceiling never activated. Both are implemented '
+ 'and both are inert under the conditions tested. They should be regarded as untested rather than validated.' }]));
body.push(RUNS([{ t: 'Censored latency. ', b: true }, { t:
  'The heavy-traffic approximation diverges when offered load exceeds pool capacity. Such ticks are censored at a finite '
+ 'ceiling, which compresses the upper tail of the latency distribution and makes the reported mean latency a lower '
+ 'bound in saturated conditions.' }]));
body.push(RUNS([{ t: 'Single service. ', b: true }, { t:
  'Only one replica pool is controlled. Real deployments comprise interdependent services sharing hosts, and the '
+ 'coupling between their scaling decisions is not modelled.' }]));
body.push(RUNS([{ t: 'Baseline fidelity. ', b: true }, { t:
  'The relaxed-and-rounded controller reproduces the integrality choice made by published optimisation-based '
+ 'auto-scalers on an otherwise identical objective; it is not a reimplementation of any specific published system, and '
+ 'it omits the chance constraints that some of them use. It should be read as an ablation of the integrality '
+ 'constraint rather than as a competing system.' }]));

// ===== VII. Conclusion
body.push(H1('VII. Conclusion'));
body.push(P(
  'Two properties of predictive container auto-scaling were measured on a simulated pool, with the predictor held at two '
+ 'extremes so that the control law was the only variable.'));
body.push(P(
  `The forecast horizon must cover the actuation dead time. Across the ${cliffDrop.length} cold-start settings sampling both sides of the boundary, the violation rate fell from `
+ `${f(meanBelow)} % to ${f(meanAbove)} % as the horizon crossed HΔt = τc, and below that boundary the predictive controller performed `
+ `worse than a reactive baseline by up to ${f(Math.abs(worstNeg))} percentage points. Horizon selection is therefore not a tuning preference `
+ 'but a feasibility condition, and it can be checked before deployment from a single measurement of cold-start latency.'));
body.push(P(
  `Relaxing the integrality of the scaling decision changes behaviour measurably. The relaxed controller disagreed with `
+ `the exact one on ${f(E3R[0].dis, 1)} to ${f(E3R[4].dis, 1)} % of control ticks, issued up to ${(E3R[4].ra / E3R[4].ea).toFixed(1)} times as many scaling actions, and `
+ `degraded from ${f(fr('rounded-oracle', 0.50).sla_mean, 2)} % to ${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations as the utilisation target was raised, against `
+ `${f(fr('pact-oracle', 0.50).sla_mean, 2)} % to ${f(fr('pact-oracle', 0.85).sla_mean, 2)} % for the exact controller. Since exact solution cost under `
+ `${f(solveMax, 2)} ms per tick, the relaxation has no performance justification at these problem sizes.`));
body.push(P(
  'The results also qualify the approach. At the default operating point the exact controller did not significantly '
+ 'reduce violations relative to either the rounded or the reactive baseline, and it carried more capacity than the '
+ 'rounded controller to achieve the service level it did. Its measured advantages are in actuation stability and in '
+ 'graceful degradation at aggressive operating points. Two of its components, the adaptive margin and the safety '
+ 'ceiling, were inert throughout.'));
body.push(P(
  'The immediate priorities for further work follow from Section VI: reproducing the horizon–dead-time transition on a '
+ 'real Docker deployment with a measured cold-start interval, repeating the comparison with a trained forecaster in '
+ 'place of the brackets, and designing a corruption model with sustained bias that would actually exercise the margin '
+ 'adaptation and the safety ceiling.'));

// ===== References
body.push(H1('References'));
const refs = [
  'T. Van Do et al., "Properties of horizontal pod autoscaling algorithms and application for scaling cloud-native network functions," 2025.',
  'B. Jeong and Y.-S. Jeong, "ARAScaler: Adaptive resource autoscaling scheme using ETimeMixer for efficient cloud-native computing," IEEE Trans. Services Computing, vol. 18, no. 1, pp. 72–84, 2025.',
  'J. W. Fog, J. J. T. Møller, T. M. Jensen, D. Taibi, and M. Albano, "Comparing neural and statistical time-series models for proactive auto-scaling in Kubernetes," in Proc. IEEE SOSE, 2025, pp. 151–161.',
  'L. Baresi, S. Guinea, A. Leva, and G. Quattrocchi, "Combined vertical and horizontal autoscaling through model predictive control," in Proc. Euro-Par, 2018, pp. 147–159.',
  'G. Quattrocchi, E. Incerto, R. Pinciroli, C. Trubiani, and L. Baresi, "Autoscaling solutions for cloud applications under dynamic workloads," IEEE Trans. Services Computing, vol. 17, no. 3, pp. 804–820, 2024.',
  'D. Zhang et al., "OptScaler: A collaborative framework for robust autoscaling in the cloud," Proc. VLDB Endowment, vol. 17, no. 12, pp. 4090–4103, 2024.',
  'Z. Cai, H. Wu, X. Jiang, X. Li, and R. Buyya, "Deep learning and feedback control based container auto-scaling for cloud native microservices," IEEE Trans. Services Computing, 2025.',
  'F. Meng, H. Dai, G. Cong, B. Zhu, and H. Zhao, "CATScaler: A convolution-augmented transformer scaling framework for cloud-native applications," IEEE Trans. Services Computing, vol. 18, no. 5, pp. 2659–2672, 2025.',
  'Y. Ding, C. Li, Z. Cai, X. Wang, and B. Yang, "A dynamic interval auto-scaling optimization method based on Informer time series prediction," IEEE Access, vol. 13, pp. 14572–14583, 2025.',
  'Z. Fang, H. Ma, G. Chen, and R. Buyya, "HGraphScale: Hierarchical graph learning for autoscaling microservice applications in container-based cloud computing," IEEE Trans. Services Computing, vol. 19, no. 1, pp. 410–422, 2026.',
  'P. Habibi, S. Rawat, and A. Leon-Garcia, "AgileSphere: Predictive-reactive Kubernetes cluster autoscaling with warm-pool node promotion," IEEE Access, vol. 14, pp. 129559–129579, 2026.',
  'B. Kumar, A. Verma, and R. Buyya, "Predictive autoscaling in cloud-native and federated cloud-edge computing environments: A taxonomy and future directions," arXiv:2606.07046, 2026.',
];
refs.forEach((r, i) => body.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: 210 },
  indent: { left: convertInchesToTwip(0.25), hanging: convertInchesToTwip(0.25) },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, size: 17, font: F })],
})));

const archWide = [
  img('fig_architecture.png', 6.95),
  CAP('Fig. 1.\u2003Implemented pipeline. Each block names the repository module that realises it. '
    + 'Dashed outlines mark components that exist in the repository but were not exercised in this '
    + 'evaluation: the learned forecaster (PyTorch unavailable) and the Docker actuation backend '
    + '(no daemon available). Bold labels are component names.'),
];

// ---------------------------------------------------------------- document
const PAGE = {
  size: { width: 12240, height: 15840 },
  margin: { top: 1150, bottom: 1150, left: 1080, right: 1080 },
};
const foot = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, font: F })],
  })],
});

const doc = new Document({
  numbering: { config: [{
    reference: 'b',
    levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
               alignment: AlignmentType.LEFT,
               style: { paragraph: { indent: { left: convertInchesToTwip(0.26),
                                               hanging: convertInchesToTwip(0.17) } } } }],
  }] },
  styles: { default: { document: { run: { font: F, size: 19 } } } },
  sections: [
    { properties: { page: PAGE, column: { count: 1 } },
      footers: { default: foot }, children: title },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE,
                    column: { count: 2, space: 400, equalWidth: true } },
      footers: { default: foot }, children: body.slice(0, SPLIT) },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE,
                    column: { count: 1 } },
      footers: { default: foot }, children: archWide },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE,
                    column: { count: 2, space: 400, equalWidth: true } },
      footers: { default: foot }, children: body.slice(SPLIT) },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(ROOT, 'PACT_paper.docx');
  fs.writeFileSync(out, buf);
  console.log('written', out, buf.length, 'bytes');
});
