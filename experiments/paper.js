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
const SEC = require('./sections.js');
const SEC2 = require('./sections2.js');
const SEC3 = require('./sections3.js');
const SEC4 = require('./sections4.js');
const E22_EXISTS = SEC4.E22_EXISTS;
const lawTestable = Object.values(SEC4.CL).filter(c => c.reactiveViolates);
const lawReal = lawTestable.length > 0 && lawTestable.every(c => c.drops === c.both.length);
const { P, RUNS, H1, H2, H3, CAP, BUL, EQI, sub, sup, img, mkTable,
        E1, E2, E3, E4, E5, E6, E7, E15, MAN, by, f, F,
        TEXT_W, COL_W, COL_GAP, E16, E17B, TCN, WC, WCT, PROV, fams, uniq, REAL_FAMS } = B;

const ROOT = path.resolve(__dirname, '..');

// ---------------------------------------------------------------- data
const po = by(E1, 'method', 'pact-oracle');
const pp = by(E1, 'method', 'pact-persist');
const ro = by(E1, 'method', 'rounded-oracle');
const rp = by(E1, 'method', 'rounded-persist');
const re = by(E1, 'method', 'reactive');
const st = by(E1, 'method', 'static');

const sig = (m, k) => E2.find(r => r.method === m && r.metric === k);
const FAM_E3 = fams(uniq(E3));
const series = a => (a.length < 2 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);
const FAM_E1 = fams(uniq(B.csv('e1_runs.csv')));
const FAM_E4 = fams(uniq(B.csv('e4_ablation_runs.csv')));
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
const E7on = E7.filter(r => r.forecast_noise_sigma >= 0 && TRUE(r.safety_guard));
const E7off = sg => E7.find(r => r.forecast_noise_sigma === sg && !TRUE(r.safety_guard));
const E7ovr = E7on.filter(r => r.overrides_total > 0);
const E7slaSame = E7on.every(r => r.sla_mean === E7off(r.forecast_noise_sigma).sla_mean);

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
      + `This paper examines both choices experimentally on a discrete-event pool simulator driven by five synthetic workload families `
      + `and by ${PROV.worldcup98.subset.days[1] - PROV.worldcup98.subset.days[0] + 1} days of the 1998 World Cup web-server log at two scalings. Every controller is run with two bracketing `
      + 'predictors, perfect foresight and persistence, so that controller effects are separated from predictor quality, and on the '
      + 'recorded traffic also with a temporal convolutional forecaster trained on its earlier days. '
      + 'Four findings follow. First, the forecast horizon must cover the actuation dead time or prediction is actively harmful: '
      + `averaged over the ${cliffDrop.length} cold-start settings for which both sides of the boundary were sampled, violations fall from ${f(meanBelow)} % to ${f(meanAbove)} % as the horizon crosses `
      + `HΔt = τc, and below that threshold the predictive controller is up to ${f(Math.abs(worstNeg))} percentage points worse than a reactive baseline, `
      + `while above it it is up to ${f(bestPos)} points better${lawReal ? '; the transition reappears on the recorded traffic'
        : lawTestable.length ? '' : '; on the recorded traffic at the scalings used, the reactive baseline never violated, so the transition could not be tested there'}. `
      + 'Second, the integer problem can be solved to global optimality rather than approximated. The dead time '
      + 'confines the tracking-relevant decisions to the first H − τ + 1 steps of the horizon and leaves the '
      + 'remainder separable. '
      + `This cuts the search from (2Δmax+1)^H to ${E15.paper_config.m} nested integer choices, and the resulting solver matched `
      + `exhaustive enumeration in ${E15.validation.exact_attains_optimum} of ${E15.validation.cases} brute-forced plant states; completing the horizon greedily `
      + `instead returned an optimal first action in only ${E15.validation.greedy_first_action_optimal} of them, and never a better one. `
      + `Third, relaxing integrality is not free: the relaxed-and-rounded `
      + `controller disagrees with the exact solution on ${f(E3R[0].dis, 1)}–${f(E3R[4].dis, 1)} % of control ticks and, at an aggressive `
      + `utilisation target of 0.85, reaches ${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations against ${f(fr('pact-oracle', 0.85).sla_mean, 2)} % for the exact controller. `
      + `Exact solution costs at most ${f(solveMax, 1)} ms per tick in the mean and ${f(E15.paper_config.exact_max_ms, 1)} ms in the worst single solve observed, under ${f(E15.paper_config.exact_max_ms / 50.0, 2)} % of the control interval. `
      + `Fourth, the trained forecaster does not sit between the brackets as the protocol assumes: it lies inside a non-degenerate bracket on `
      + `${SEC4.nBetween - SEC4.degenerate.length} of ${SEC4.nBetween + SEC4.nBelow + SEC4.nAbove - SEC4.degenerate.length} closed-loop comparisons. Its upward forecast bias reproduces the direction of its capacity-side effects in `
      + `${SEC4.capAgree} of ${SEC4.capRows.length} cases but not its reduction of the rounded controller’s churn. `
      + 'These results come from simulation, not from a deployed system; every value reported is measured, and none is '
      + 'estimated or illustrative.' }),
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
  'Three gaps motivate this work, and all three concern the controller rather than the predictor.'));
body.push(P(
  'The first is integrality. Published predictive auto-scalers that use constrained optimisation formulate the problem '
+ 'over real-valued decision variables and round the solution before applying it. The rounding step is reported as an '
+ 'implementation detail and its cost is not measured. Independent evidence suggests the cost is not negligible: in a '
+ 'recent benchmark of reinforcement-learning controllers, discrete-action algorithms outperformed continuous-action '
+ 'ones, an outcome the authors attribute to the mismatch introduced when continuous outputs are bucketed into integer '
+ 'replica changes. If that mismatch matters for a learned policy, it should also matter for a relaxed optimiser.'));
body.push(P(
  'The second is tractability, and it is the reason the first gap has stayed open. Enforcing integrality on a '
+ 'receding-horizon problem appears to demand a search over (2Δmax + 1) raised to the power of the horizon length, '
+ 'which is why the usual response is either to relax the constraint or to enumerate the first action and complete '
+ 'the horizon with a heuristic. The second response is often described as exact on the grounds that only the first '
+ 'action is applied. That reasoning does not hold: the first action is chosen by comparing the costs of completed '
+ 'plans, so a biased completion biases the comparison. Whether an exact integer solve is affordable at '
+ 'container-scaling rates has not, as far as this survey found, been settled either way.'));
body.push(P(
  'The third is horizon selection. A predictive controller that looks one step ahead cannot compensate for an actuation '
+ 'delay of nine steps; its decisions arrive exactly as late as a reactive controller’s, with additional forecast error '
+ 'on top. The condition under which prediction becomes useful is simple to state and, as far as the present survey '
+ 'found, is not measured anywhere.'));

SEC2.workedExample.forEach(e => body.push(e));

body.push(H2('C. Contributions'));
body.push(P('This paper makes five contributions, each supported by measurements reported in Section V.'));
body.push(BUL(
  'A measurement of the horizon–dead-time law. Sweeping the cold-start interval and the forecast horizon independently '
+ 'shows a sharp transition in service-level attainment that coincides with the condition HΔt ≥ τc. Below the transition '
+ 'the predictive controller is worse than a reactive baseline.'));
body.push(BUL(
  'A tractable exact solve, with its structural argument and its verification. Because an action issued now does '
+ 'not reach the pool for τ ticks, only the first H − τ + 1 actions of the horizon can influence any tracking term '
+ 'inside it, and the remaining actions enter the cost through a separable chain minimised by backward recursion. '
+ 'Together these cut the search to a size that can be enumerated, and the resulting solver is checked against '
+ 'exhaustive enumeration rather than asserted to be correct.'));
body.push(BUL(
  'A measurement of the cost of relaxation, and of the cost of approximating the solve. The exact controller is '
+ 'compared against an otherwise identical relaxed-and-rounded controller across five pool sizes, isolating the '
+ 'effect of the integrality constraint alone, and separately against a solver that enumerates first actions under '
+ 'a greedy horizon completion, isolating the effect of solving the integer problem only approximately.'));
body.push(BUL(
  'An evaluation protocol that separates controller effects from predictor quality by bracketing every controller between '
+ 'a perfect-foresight predictor and a persistence predictor, so that conclusions holding under both brackets are '
+ 'attributable to the control law.'));
body.push(BUL(
  'A test of the protocol and of the findings on recorded traffic. A temporal convolutional forecaster is trained on '
+ 'ten days of the 1998 World Cup web-site access log, compared against recurrent, attention-based and statistical '
+ 'baselines, and placed in the loop between the two brackets; the earlier findings are then re-tested on the '
+ 'recorded trace, and those that do not survive are reported as not surviving.'));

body.push(H2('D. Organisation'));
body.push(P(
  'Section II reviews existing auto-scaling systems and groups the recent literature by the mechanism each study adopts. '
+ 'Section III develops the controller. Section IV describes the experimental setup, the simulated plant and the workloads. '
+ 'Section V reports results, ending in Section V-Q with the recorded trace and the trained forecaster. Section VI '
+ 'discusses them, Section VII states the limitations, several of which are substantial, and Section VIII concludes.'));

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
  'A smaller body of work treats scaling as a control problem. Incerto et al. (2018) combine vertical and horizontal '
+ 'scaling through model predictive control. Quattrocchi et al. (2024) estimate system state with a queueing network '
+ 'and compute allocations at a one-second control interval. Zou et al. (2024), in the OptScaler framework deployed at '
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
SEC.litSection.forEach(e => body.push(e));

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
body.push(EQI(1));
body.push(P('A forecast of utilisation is converted to a replica requirement through the utilisation law (2), in which '
+ 'u* is the target operating point, μ the per-replica service rate, and γ a safety margin. The ceiling is used because '
+ 'a fractional shortfall is a real shortfall.'));
body.push(EQI(2));
body.push(P('Memory imposes an independent bound, so the binding requirement is the larger of the two, as in (3).'));
body.push(EQI(3));
body.push(P('Expected tail latency at a candidate replica count is estimated by the heavy-traffic approximation (4), '
+ 'where s̄ is the mean service time and cₐ² and cₛ² are the squared coefficients of variation of the arrival and '
+ 'service processes. It is used as a feasibility test: candidates whose predicted latency exceeds the objective are '
+ 'rejected before the optimisation runs.'));
body.push(EQI(4));

body.push(H2('B. Plant Model and Dead Time'));
body.push(P('The pool evolves according to (5): the replica count changes by the control input applied τ ticks earlier, '
+ 'where τ is the cold-start interval expressed in control ticks. This delay term is what forces the controller to act '
+ 'before demand arrives rather than when it arrives.'));
body.push(EQI(5));
body.push(P('For the forecast to be actionable it must reach at least τ ticks into the future, which gives the '
+ 'feasibility condition (6). Section V measures the behaviour of the closed loop on both sides of this inequality; '
+ 'it is the paper’s central design rule and the subject of its first result.'));
body.push(EQI(6));

body.push(H2('C. Receding-Horizon Objective'));
body.push(P('The applied action minimises (7) over the horizon. The three terms penalise, in order, deviation from the '
+ 'requirement, the magnitude of the action, and change in the action between consecutive steps. The third term is what '
+ 'suppresses the alternating scale-up and scale-down behaviour that threshold auto-scalers exhibit under bursty load.'));
body.push(EQI(7));
body.push(P('The tracking weight is asymmetric, as in (8): being short of the requirement is penalised more heavily than '
+ 'being over it, because a dropped request costs more than an idle replica-second.'));
body.push(EQI(8));
body.push(P('The optimisation is subject to (9): the pool is bounded, the per-tick change is rate limited, and the '
+ 'decision is an integer. The last of these is the constraint that related work relaxes.'));
body.push(EQI(9));
body.push(P('The integer constraint is kept rather than relaxed. Section III-E shows that the actuation dead time '
+ 'confines the decisions that can affect the horizon to its first few steps and renders the remainder separable, which '
+ 'is what makes an exact solve affordable here. Section V reports the cost of this choice '
+ `in both directions: the solve time, which stays below ${f(E15.paper_config.exact_max_ms, 1)} ms per tick, and the decisions it changes.`));
body.push(P('Only the first element of the optimal sequence is applied and the problem is re-solved at the next tick. '
+ 'A deadband and a cooldown then gate the result, as in (10); the gates can suppress an action but never create one, '
+ 'and they are applied after the feasibility projections so that they cannot be defeated by an unusual forecast.'));
body.push(EQI(10));

body.push(H2('D. Margin Adaptation and Safety Ceiling'));
body.push(P('A drift monitor compares the replica count the observed load actually required against the count predicted '
+ 'τ ticks earlier and adapts the margin by the clipped integral update (11). A separate ceiling bypasses the optimiser '
+ 'and every gate when observed utilisation exceeds an emergency threshold, so that an unprecedented burst is not held '
+ 'back by the deadband. Sections V-E, V-F and V-Q report when each of them acted.'));
body.push(EQI(11));

// ===== IV. Setup
SEC.algos.forEach(e => body.push(e));

SEC3.moduleDetail.forEach(e => body.push(e));

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
  { t: `All runs executed on ${MAN.platform}, Python ${MAN.python}, NumPy ${MAN.numpy}, PyTorch ${MAN.torch} and `
     + `statsmodels ${MAN.statsmodels}, at repository commit ${String(MAN.git_sha).slice(0, 10)}`
     + `${MAN.git_dirty ? ' with uncommitted changes' : ''}. The neural forecasters were trained on the `
     + `${String(TCN.train_device).toUpperCase()} device of that machine and evaluated on its CPU. Each run is `
     + `${MAN.n_ticks_per_run} control ticks at Δt = 5 s, giving a simulated duration of `
     + `${(MAN.n_ticks_per_run * 5 / 60).toFixed(0)} minutes, and starts with four replicas or the requirement (2) of its `
     + `first tick, whichever is larger. Results are reported over ${MAN.n_seeds} seeds unless stated otherwise.` },
]));

SEC3.plantDetail.forEach(e => body.push(e));

body.push(H2('C. Datasets'));
body.push(P(
  'Two kinds of arrival trace are used. Five synthetic families are generated by the harness in the repository, chosen '
+ 'to span distinct demand shapes rather than to sample one shape repeatedly. Each is parameterised by a seed that '
+ 'perturbs it with additive Gaussian noise, and the same seed produces the same trace for every method, so comparisons '
+ `are paired. The recorded workload is the 1998 World Cup web-site access log [11], days ${PROV.worldcup98.subset.days[0]}–`
+ `${PROV.worldcup98.subset.days[1]}, converted to an arrival rate at the control interval and rescaled at two levels so `
+ 'that it fits the simulated pool; Section V-Q describes the trace, the scaling and the chronological split. For a '
+ 'recorded family the seed selects one of the disjoint windows of the test split, so recorded runs are paired in the '
+ 'same way.'));
body.push(mkTable(
  ['Trace family', 'Shape', 'Provenance', 'Ticks per run', 'Seeds'],
  [
    ['Diurnal', 'Sinusoidal base, period 300 ticks', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Burst', 'Two Gaussian bursts on a flat base', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Ramp', 'Linear rise, plateau, linear fall', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Step', 'Single discontinuity at mid-run', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['Sawtooth', 'Repeating linear rise, period 90 ticks', 'Synthetic (this work)', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['WC98/trace', `Recorded; whole trace peaks at ${f(100 * WC.peak_fraction_of_capacity, 0)} % of capacity`, 'WorldCup98 [11]', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
    ['WC98/test', `Recorded; test split peaks at ${f(100 * WCT.peak_fraction_of_capacity, 0)} % of capacity`, 'WorldCup98 [11]', String(MAN.n_ticks_per_run), String(MAN.n_seeds)],
  ],
  [1700, 2600, 1900, 1100, 800], { left: true }));
body.push(CAP('Table II. Workload families. The synthetic traces are generated by experiments/harness.py; the '
+ 'recorded ones are windows of the WorldCup98 test split, produced by experiments/make_real_traces.py from the '
+ 'archived logs. Seeds gives the count used in the main comparison; other experiments state their own. Every '
+ 'method sees the identical trace for a given family and seed.'));

body.push(H2('D. Forecasters and Bracketing'));
body.push(P(
  'The controller consumes a forecast of utilisation and memory over its horizon, and three predictors supply it. The '
+ 'oracle has perfect foresight of the arrival series and represents the limit of what forecasting can contribute. '
+ 'Persistence repeats the most recent observation across the horizon and represents the absence of a model. These '
+ 'two brackets drive every experiment, so that a conclusion holding under both can be attributed to the control law '
+ 'rather than to the predictor.'));
body.push(P(
  `The third predictor is the temporal convolutional network of the forecasting module, trained on the arrival rate of `
+ `the recorded trace and frozen before any closed-loop run (Section V-Q). Its ${TCN.horizon}-step arrival forecast is converted `
+ 'to utilisation and memory by the same capacity law that converts the oracle’s, so the three predictors differ only '
+ 'in the arrival series they supply. The trained arm is used only on recorded traffic, where there is a training split '
+ 'to learn from, and not in the horizon sweeps, because the checkpoint’s horizon is fixed. As a forecaster it is '
+ 'compared against LSTM, GRU, bidirectional LSTM and Transformer baselines of matched size, an ARIMA model selected '
+ 'by AIC, and persistence, all scored on the same test origins.'));

body.push(H2('E. Methods Compared'));
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

body.push(H2('F. Metrics and Statistics'));
body.push(P(
  'Service-level violation rate is the fraction of ticks whose tail latency exceeds the objective. Over-provisioning '
+ 'ratio is the mean positive excess of the replica count over the requirement. Actuation churn is the number of ticks '
+ 'on which the replica count changed. Cost is replica-seconds per thousand requests served. Comparisons against the '
+ 'proposed controller use the Wilcoxon signed-rank test on paired runs, with Holm correction across the family of '
+ 'comparisons and Cliff’s delta as an effect size; confidence intervals are bias-corrected and accelerated bootstrap '
+ 'intervals over 4000 resamples. Significance is never claimed without the corrected p-value.'));

// ===== V. Results
SEC2.configTable.forEach(e => body.push(e));

body.push(H1('V. Results and Discussion'));
body.push(P(
  `All values in this section are computed from the run logs written to the repository’s results directory. Every `
+ `table and figure from here on is generated directly from those files.`));

body.push(H2('A. The Horizon–Dead-Time Law'));
body.push(P(
  'The first experiment sweeps the cold-start interval τc over six values and the forecast horizon H over ten, and '
+ 'measures the resulting violation rate for the proposed controller and for the reactive baseline. Fig. 2 shows the '
+ 'outcome as a heat map with the boundary HΔt = τc from (6) drawn on it, and Fig. 3 shows the same data as violation '
+ 'against lookahead for each cold-start setting.'));
body.push(P(
  'The transition is sharp and it coincides with the predicted boundary. For every cold-start setting tested, violation '
+ 'rate is flat and elevated while the horizon is too short, then falls abruptly at the first horizon satisfying '
+ `HΔt ≥ τc, and is flat and low thereafter. The grid covers six cold-start values, but at the shortest of them every horizon tested already satisfies the bound, so that setting yields no below-boundary measurement and is excluded from the average. Over the remaining ${cliffDrop.length}, which bracket the boundary on both sides, the rate falls from ${f(meanBelow)} % below `
+ `the boundary to ${f(meanAbove)} % above it. The effect is monotone in τc: at τc = ${cliff[0].tau} s the controller is already `
+ `near its floor at the shortest horizon tested, whereas at τc = ${cliff[cliff.length - 1].tau} s it remains at `
+ `${f(cliff[cliff.length - 1].belowMean)} % until the horizon reaches ${cliff[cliff.length - 1].firstH} ticks.`));
body.push(P(
  'The comparison against the reactive baseline is the operationally important part, and it is not flattering to naive '
+ 'predictive scaling. When the horizon does not cover the dead time the predictive controller is not merely no better '
+ `than reacting; it is worse, by up to ${f(Math.abs(worstNeg))} percentage points. Once the horizon covers the dead time it is better, by up `
+ `to ${f(bestPos)} points. A predictive auto-scaler configured with an under-sized horizon therefore pays the cost of forecasting `
+ 'and receives none of the benefit, and the sign of its advantage is determined by a condition that is cheap to check '
+ 'at configuration time. What makes the condition useful is that it can be checked before deployment: it compares the configured horizon against the measured ready-time of the image, and needs no model of the workload.'));
body.push(img('fig_horizon_law.png', 3.35));
body.push(CAP('Fig. 2. Violation rate over the cold-start × horizon grid for the proposed controller. The solid line is '
+ 'the boundary HΔt = τc from (6). Synthetic traces; mean of three families × three seeds per cell.'));
body.push(img('fig_horizon_cliff.png', 3.35));
body.push(CAP('Fig. 3. The same data as violation against lookahead. Dotted verticals mark τc for each curve. '
+ 'Synthetic traces.'));
SEC2.cliffTable.forEach(e => body.push(e));

body.push(H2('B. The Cost of Relaxing Integrality'));
body.push(P(
  'The second experiment compares the integer-exact controller against the relaxed-and-rounded controller across five '
+ 'pool bounds, holding everything else fixed. Table V reports the result and Fig. 4 shows it graphically.'));
body.push(mkTable(
  ['Nₘₐₓ', 'Ticks where\ndecisions differ (%)', 'Churn, exact', 'Churn, rounded', 'Δ violation (pp)', 'Mean solve (ms)'],
  E3R.map(r => [String(r.n), f(r.dis, 1), f(r.ea, 1), f(r.ra, 1), f(r.dsla, 2), f(r.solve, 3)]),
  [900, 2100, 1500, 1600, 1500, 1500]));
body.push(CAP(`Table V. Integer-exact against relaxed-and-rounded control, by pool bound. ${FAM_E3}, `
+ `${E3.filter(r => r.n_max === 2).length / uniq(E3).length} seeds each, pooled per row. A positive Δ violation means the rounded controller violated more often.`));
const dslaLo = Math.min(...E3R.map(r => r.dsla)), dslaHi = Math.max(...E3R.map(r => r.dsla));
const dslaArg = E3R.find(r => r.dsla === dslaHi);
const fewer = E3R.filter(r => r.ea < r.ra), more = E3R.filter(r => r.ea >= r.ra);
const orders = Math.floor(Math.log10(5000 / solveMax));
body.push(P(
  `The two controllers disagree on a substantial and increasing fraction of ticks, from ${f(E3R[0].dis, 1)} % at the smallest `
+ `pool to ${f(E3R[4].dis, 1)} % at the largest. Pooled over families, the rounded controller violates the service level more `
+ `often at ${E3R.filter(r => r.dsla > 0).length} of the ${E3R.length} pool bounds, by between ${f(dslaLo, 2)} and ${f(dslaHi, 2)} percentage points; `
+ `the largest gap is at Nₘₐₓ = ${dslaArg.n}, where one replica is a large share of the pool. Churn depends on the pool bound. `
+ `At Nₘₐₓ = 16 the exact controller issues ${f(E3R[3].ea, 0)} scaling actions per run against ${f(E3R[3].ra, 0)} for the rounded controller, and at `
+ `Nₘₐₓ = 32 the figures are ${f(E3R[4].ea, 0)} against ${f(E3R[4].ra, 0)}. Rounding converts a fractional preference into a `
+ 'full replica change, and because the fractional part fluctuates from tick to tick the rounded controller oscillates '
+ `where the exact controller holds. ${fewer.length === E3R.length ? 'This holds at every pool bound.' : `It does not hold at Nₘₐₓ = ${series(more.map(r => r.n))}, `
+ `where the rounded controller issues fewer actions (${more.map(r => `${f(r.ra, 0)} against ${f(r.ea, 0)}`).join('; ')}); this experiment `
+ 'does not isolate why, so the churn advantage is claimed only for the larger pools.'}`));
body.push(P(
  `The cost of solving exactly is small. The mean solve time is at most ${f(solveMax, 2)} ms per tick, at the largest pool `
+ `tested, which is ${orders === 1 ? 'one order' : `${['zero', 'one', 'two', 'three', 'four'][orders]} orders`} of magnitude inside the 5 s control interval. `
+ 'This experiment, the solve-time sweep of Section V-H and the solver comparison of Section V-I were timed with no '
+ 'other experiment running on the machine. '
+ 'On this evidence the integrality constraint should not be relaxed for performance reasons at these problem sizes.'));
body.push(img('fig_rounding_gap.png', 6.9));
body.push(CAP('Fig. 4. Disagreement between the exact and rounded controllers, and the resulting actuation churn. '
+ `${FAM_E3}.`));

body.push(H2('C. Operating-Point Sensitivity'));
body.push(P(
  'Comparing controllers at a single target utilisation is misleading, because each sits at a different point on the '
+ 'trade-off between service level and capacity. The third experiment therefore sweeps the target from 0.50 to 0.85 and '
+ 'traces the attainable frontier for each method. Table VI reports the extremes and Fig. 5 shows the full sweep.'));
body.push(mkTable(
  ['Target u*', 'Exact MPC\nviol. (%)', 'Exact MPC\nover (%)', 'Rounded\nviol. (%)', 'Rounded\nover (%)', 'Reactive\nviol. (%)', 'Reactive\nover (%)'],
  [0.50, 0.65, 0.75, 0.85].map(u => [
    f(u, 2),
    f(fr('pact-oracle', u).sla_mean, 2), f(fr('pact-oracle', u).over_mean, 1),
    f(fr('rounded-oracle', u).sla_mean, 2), f(fr('rounded-oracle', u).over_mean, 1),
    f(fr('reactive', u).sla_mean, 2), f(fr('reactive', u).over_mean, 1),
  ]),
  [1100, 1400, 1400, 1300, 1300, 1300, 1300]));
body.push(CAP(`Table VI. Service level and capacity cost against target utilisation. ${fams(MAN.workloads)}, `
+ `${fr('pact-oracle', 0.65).n_runs} runs per cell, oracle predictor.`));
body.push(P(
  'The two controllers degrade very differently as the target is pushed up. The rounded controller is efficient and '
+ `adequate at conservative targets — at u* = 0.50 it violates on ${f(fr('rounded-oracle', 0.50).sla_mean, 2)} % of ticks while carrying only `
+ `${f(fr('rounded-oracle', 0.50).over_mean, 1)} % excess capacity — but it collapses as the target rises, reaching `
+ `${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations at u* = 0.85. The exact controller rises far more slowly over the same range, from `
+ `${f(fr('pact-oracle', 0.50).sla_mean, 2)} % to ${f(fr('pact-oracle', 0.85).sla_mean, 2)} %. The mechanism is visible in Table V: when the pool is run close to `
+ 'its limit, a rounding error of one replica is a large fraction of the available headroom, and the rounded controller '
+ 'spends an increasing share of its time on the wrong side of the requirement.'));
body.push(P(
  'Neither controller dominates. The exact controller buys service level '
+ `with capacity: at u* = 0.65 it carries ${f(fr('pact-oracle', 0.65).over_mean, 1)} % excess against the rounded controller’s `
+ `${f(fr('rounded-oracle', 0.65).over_mean, 1)} %, for a violation rate of ${f(fr('pact-oracle', 0.65).sla_mean, 2)} % against `
+ `${f(fr('rounded-oracle', 0.65).sla_mean, 2)} %. An operator who can tolerate the violation rate of the rounded controller should prefer it on cost. `
+ 'The case for exact solution is that it remains usable at operating points where the rounded controller does not; '
+ 'its computational cost is reported in Section V-H.'));
body.push(img('fig_frontier.png', 3.35));
body.push(CAP(`Fig. 5. Attainable frontier as the target utilisation is swept from 0.50 to 0.85. ${fams(MAN.workloads)}, `
+ `oracle predictor. Points are labelled with the target that produced them.`));

body.push(H2('D. Aggregate Comparison and Significance'));
body.push(P(
  `Table VII reports the main comparison over ${MAN.workloads.length} workload families × ${MAN.n_seeds} seeds, with bootstrap confidence `
+ 'intervals, under both forecaster brackets. Table VIII reports the paired significance tests against the proposed '
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
body.push(CAP(`Table VII. Main comparison. ${FAM_E1}, ${MAN.n_seeds} seeds each = ${po.n_runs} runs per row. `
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
body.push(CAP('Table VIII. Paired Wilcoxon signed-rank tests against the integer-exact controller with the oracle '
+ `predictor, Holm-corrected across the comparison family. n = ${sig('reactive', 'sla').n_pairs} paired runs. ${FAM_E1}.`));
const isSig = r => r.p_holm !== '' && r.p_holm !== undefined && Number(r.p_holm) < 0.05;
const slaR = sig('rounded-oracle', 'sla'), slaH = sig('reactive', 'sla');
body.push(P(
  'Two aspects of Table VIII deserve emphasis because they qualify the paper’s claims. First, the difference in violation '
+ `rate between the proposed controller and the rounded controller at this operating point is ${isSig(slaR) ? '' : 'not '}significant after `
+ `correction (p = ${f(slaR.p_holm, 3)}), and the difference against the reactive baseline is ${isSig(slaH) ? '' : 'not '}significant either `
+ `(p = ${f(slaH.p_holm, 3)}). The service-level advantage reported in Section V-C emerges at aggressive operating `
+ 'points, not at the default one. Second, the differences that are significant are in churn and capacity: against the '
+ `rounded controller the reduction in churn has a Cliff’s delta of ${f(sig('rounded-oracle', 'act').cliffs_delta, 2)}, and against the reactive `
+ `baseline the reduction in over-provisioning has a delta of ${f(sig('reactive', 'over').cliffs_delta, 2)}. The robust claims are about stability `
+ 'and cost, not about service level in the ordinary regime.'));
const keep = k => Math.sign(po[k] - ro[k]) === Math.sign(pp[k] - rp[k]);
const ordered = ['sla_mean', 'over_mean', 'act_mean'].filter(keep);
body.push(P(
  'The forecaster brackets bound the predictor’s contribution under this protocol. Moving from perfect foresight to '
+ `persistence changes violations from ${f(po.sla_mean)} % to ${f(pp.sla_mean)} % and churn from ${f(po.act_mean, 0)} to ${f(pp.act_mean, 0)} actions for the exact `
+ `controller, an interval of ${f(Math.abs(pp.sla_mean - po.sla_mean), 2)} percentage points in violation rate, and from ${f(ro.sla_mean)} % to `
+ `${f(rp.sla_mean)} % for the rounded one. The ordering of the exact and rounded controllers is the same under both brackets `
+ `for ${ordered.length} of the 3 metrics compared (violation, over-provisioning, churn), which is the condition under which `
+ 'a difference is attributed to the control law. Section V-Q tests the further assumption that a trained forecaster '
+ 'lands inside the bracket, and finds that it often does not.'));
body.push(img('fig_trace.png', 6.9));
body.push(CAP('Fig. 6. Replica count and tail latency on one burst trace. The rounded controller overshoots at each '
+ 'burst and the reactive controller holds capacity long after demand falls. Synthetic burst trace, seed 1001, oracle '
+ 'predictor.'));
body.push(img('fig_main_comparison.png', 6.9));
body.push(CAP(`Fig. 7. Main comparison with 95 % bootstrap intervals. ${FAM_E1}, ${po.n_runs} runs per bar.`));

body.push(H2('E.\u2003Ablation'));
const E4R = B.csv('e4_ablation_runs.csv');
const runsOf = (v, real) => E4R.filter(r => r.variant === v && REAL_FAMS.includes(r.workload) === real);
const count = (rs, k) => ({ n: rs.length, on: rs.filter(r => Number(r[k]) > 0).length,
                            total: rs.reduce((s, r) => s + Number(r[k]), 0) });
const ovSyn = count(runsOf('full', false), 'overrides'), ovReal = count(runsOf('full', true), 'overrides');
const gmSyn = count(runsOf('full', false), 'gamma_max'), gmReal = count(runsOf('full', true), 'gamma_max');
const pAbl = v => abl(v).sla_p_raw;
const gmPers = count(B.csv('e1_runs.csv').filter(r => r.method === 'pact-persist'), 'gamma_max');
const nonSig = E4.filter(r => r.variant !== 'full' && Number(r.sla_p_raw) >= 0.05).length;
const nAbl = E4.filter(r => r.variant !== 'full').length;
body.push(P(
  'Each component of the controller was disabled in turn, with everything else held fixed, over '
+ `${FAM_E4.charAt(0).toLowerCase() + FAM_E4.slice(1)}, ${MAN.n_seeds} seeds each, with the oracle predictor. Table IX reports the outcome and Fig. 8 shows it. `
+ `No component changes the violation rate significantly: ${nonSig} of the ${nAbl} service-level comparisons have p ≥ 0.05 uncorrected. `
+ 'What the components change is how much capacity the controller holds and how often it moves.'));
body.push(P(
  'Removing the optimisation layer entirely, so that the controller applies the first element of the demand trajectory '
+ `directly, changes the violation rate by ${f(abl('no_mpc').sla_delta_vs_full, 2)} percentage points (p = ${f(pAbl('no_mpc'), 3)}) and reduces `
+ `over-provisioning by ${f(Math.abs(abl('no_mpc').over_delta_vs_full), 1)} points. Removing the churn penalty changes service level by `
+ `${f(abl('no_churn').sla_delta_vs_full, 2)} points but raises over-provisioning by ${f(abl('no_churn').over_delta_vs_full, 1)} points and churn by `
+ `${f(abl('no_churn').act_delta_vs_full, 0)} actions, consistent with the rounding result of Section V-B: penalising change is what keeps `
+ 'the pool steady.'));
body.push(P(
  'Disabling cold-start compensation, so that the controller plans as if a replica were ready the moment it is '
+ `requested, changes the violation rate by only ${f(abl('no_coldstart').sla_delta_vs_full, 2)} points (p = ${f(pAbl('no_coldstart'), 3)}), but it cuts `
+ `over-provisioning by ${f(Math.abs(abl('no_coldstart').over_delta_vs_full), 1)} points and churn by ${f(Math.abs(abl('no_coldstart').act_delta_vs_full), 1)} actions. `
+ 'An earlier version of this paper reported only the first of these and called the component ineffective; the '
+ 'capacity and churn effects were present in that version’s result files as well. The delay term makes the '
+ 'controller provision ahead of the dead time, and on these workloads, with a horizon that already covers the dead '
+ 'time, the measurements do not show a service-level return for the capacity it holds. This does not contradict '
+ 'Section V-A, where the horizon itself was varied.'));
body.push(P(
  `The two protective mechanisms behave differently. The adaptive margin moved in ${gmSyn.on} of ${gmSyn.n} synthetic runs `
+ `and ${gmReal.on} of ${gmReal.n} recorded runs of the full controller, and disabling it changes violations by `
+ `${f(abl('no_margin').sla_delta_vs_full, 2)} points and over-provisioning by ${f(abl('no_margin').over_delta_vs_full, 3)} points: under perfect `
+ 'foresight it is close to inert. It is not inert under persistence: in the main comparison it moved in '
+ `${gmPers.on} of ${gmPers.n} runs of the exact controller with the persistence predictor. `
+ `The safety ceiling, by contrast, is active: it issued ${ovSyn.total} overrides in ${ovSyn.on} of ${ovSyn.n} synthetic runs `
+ `and ${ovReal.total} in the ${ovReal.n} recorded runs. Removing it leaves the violation rate `
+ `${abl('no_safety').sla_delta_vs_full === 0 ? 'exactly unchanged' : `changed by ${f(abl('no_safety').sla_delta_vs_full, 2)} points`} `
+ `while lowering over-provisioning by ${f(Math.abs(abl('no_safety').over_delta_vs_full), 1)} points and churn by `
+ `${f(Math.abs(abl('no_safety').act_delta_vs_full), 1)} actions, so in these runs it bought capacity without a measured service-level `
+ 'return. An earlier version of this paper stated that the ceiling never activated; that was wrong for the main '
+ 'comparison and the ablation, and has been corrected.'));
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
body.push(CAP(`Table IX. Ablation. Full controller: violation ${f(abl('full').sla_mean, 2)} %, over-provisioning `
+ `${f(abl('full').over_mean, 1)} %, churn ${f(abl('full').act_mean, 1)}. ${FAM_E4}, ${abl('full').n_runs} runs per row, oracle predictor. `
+ `${nonSig === nAbl ? 'No service-level delta reaches p < 0.05.' : `${nAbl - nonSig} service-level delta(s) reach p < 0.05.`}`));
body.push(img('fig_ablation.png', 6.9));
body.push(CAP(`Fig. 8. Effect of disabling each component. ${FAM_E4}.`));

body.push(H2('F. Robustness to Forecast Corruption'));
body.push(P(
  'The final experiment injects multiplicative noise into the series seen by the forecaster while the plant continues to '
+ 'receive the true arrivals, sweeping the corruption magnitude and repeating the sweep with the safety ceiling disabled. '
+ 'Fig. 9 shows the result, which is counter-intuitive and worth stating carefully.'));
body.push(P(
  `Violation rate does not rise with corruption; ${noiseHi.sla_mean < noise0.sla_mean ? 'it falls slightly' : 'it stays level'}, from ${f(noise0.sla_mean, 3)} % at zero noise to `
+ `${f(noiseHi.sla_mean, 3)} % at σ = 0.4, while over-provisioning rises from ${f(noise0.over_mean, 1)} % to ${f(noiseHi.over_mean, 1)} %. `
+ 'The explanation is the asymmetric tracking weight of (8). Multiplicative noise causes the forecast to over-predict as '
+ 'often as it under-predicts, but the controller responds to over-prediction by adding capacity and to under-prediction '
+ 'by removing it, and the asymmetric penalty makes the former cheaper. Noise therefore acts as an accidental safety '
+ 'margin, purchased with capacity. This is a property of the cost function rather than a desirable robustness result, '
+ 'and it means the controller is not stress-tested by symmetric noise; a corruption model with sustained negative bias '
+ 'would be a sharper test and is left for future work.'));
body.push(P(
  `${E7ovr.length ? `The safety ceiling issued overrides at ${E7ovr.length} of the ${E7on.length} corruption levels, ${E7ovr.reduce((a, r) => a + r.overrides_total, 0)} in total, `
  + `but the violation rate with and without it is ${E7slaSame ? 'identical at every level' : 'different at some levels'}; only over-provisioning moves, by at most `
  + `${f(Math.max(...E7on.map(r => Math.abs(r.over_mean - E7off(r.forecast_noise_sigma).over_mean))), 2)} points.` : 'The safety ceiling never activated at any corruption level, so the two curves coincide exactly.'} `
+ `The reactive baseline, measured on the same traces, violates on ${f(reRef.sla_mean, 2)}\u202f% of ticks, `
+ `${E7on.every(r => r.sla_mean < reRef.sla_mean) ? 'above every point of the swept range' : 'inside the swept range'}. Symmetric noise does not `
+ 'produce the sustained under-prediction the ceiling exists to catch, so this experiment does not show what it is worth.'));
body.push(img('fig_robustness.png', 3.35));
body.push(CAP('Fig. 9. Violation rate against forecast corruption, with and without the safety ceiling. Synthetic '
+ `traces (burst, step and diurnal), ${noise0.n_runs} runs per point.`));

// ===== VI. Limitations
SEC.sensitivity.forEach(e => body.push(e));        // V-G
SEC.solvetime.forEach(e => body.push(e));          // V-H
SEC.exactGap.forEach(e => body.push(e));           // V-I
SEC.coldstart.forEach(e => body.push(e));          // V-J
SEC.perWorkload.forEach(e => body.push(e));        // V-K
SEC.decisions.forEach(e => body.push(e));          // V-L
SEC2.predictorQuality.forEach(e => body.push(e));  // V-M
SEC.guidance.forEach(e => body.push(e));           // V-N
SEC3.inferenceTrace.forEach(e => body.push(e));    // V-O
SEC2.positioning.forEach(e => body.push(e));       // V-P
SEC4.section.forEach(e => body.push(e));           // V-Q
SEC2.discussion.forEach(e => body.push(e));        // VI

body.push(H1('VII. Limitations and Threats to Validity'));
SEC3.validity.forEach(e => body.push(e));
body.push(H2('D. Enumerated Limitations'));
body.push(P(
  'This evaluation has substantial limitations. They are listed here rather than distributed through the text because '
+ 'several of them bound the claims directly.'));
body.push(RUNS([{ t: 'Simulated plant only. ', b: true }, { t:
  'No result in this paper was obtained on real containers. The plant is the analytical queueing simulator described in '
+ 'Sections IV-A and IV-B, and the cold-start interval is a parameter of that simulator rather than a measurement. Image pull '
+ 'variance, scheduler interference, connection draining and load-balancer registration delay are all absent. The '
+ 'horizon–dead-time law of Section V-A is a property of the model as much as of the controller, and confirming it on a '
+ 'real Docker deployment is the single most important piece of outstanding work.' }]));
body.push(RUNS([{ t: 'One recorded trace, rescaled. ', b: true }, { t:
  `Five of the seven workload families are generated rather than recorded. The two recorded families are one trace, `
+ `days ${PROV.worldcup98.subset.days[0]}–${PROV.worldcup98.subset.days[1]} of the WorldCup98 log, at two scalings, and a scaling preserves the shape of the traffic `
+ 'but not its absolute level: real request rates were divided by a constant to fit a sixteen-replica pool, so the '
+ 'per-replica variability is not that of the original servers. The second scaling was chosen from the peak of the '
+ 'test split, which uses one statistic of the test data; the forecaster’s inputs are standardised with training '
+ 'statistics, under which a constant factor cancels, but the choice of operating point is not blind. The test split '
+ 'is a single weekend, the trace contains a short dip at each local midnight that was left unrepaired, and a second '
+ 'recorded source (Bitbrains GWA-T-12) was located but not used because it records per-machine utilisation at '
+ 'five-minute intervals rather than request arrivals.' }]));
body.push(RUNS([{ t: 'The start state was changed after a diagnosis. ', b: true }, { t:
  'Runs originally opened with four replicas. On recorded windows that placed every violation in the start-up '
+ 'transient, so the rule was changed to the larger of four and the first tick’s requirement and every experiment '
+ 'was re-run (Section V-Q). The change was made before any trained-arm result existed and applies to every '
+ 'workload, but it is a change of protocol made after looking at data, and it is reported as one.' }]));
body.push(RUNS([{ t: 'One forecaster, trained once. ', b: true }, { t:
  `The trained forecaster is a single TCN checkpoint from one seed, and each baseline was also trained once, so the `
+ `ranking in Table XIX carries no estimate of training variance. The TCN ranked ${SEC4.tcnRank} of ${SEC4.atH.length} on accuracy at the `
+ `full horizon, behind ${SEC4.MNAME[SEC4.best.m]}; a better-tuned model might change where the trained arm falls relative to `
+ `the brackets. It was trained on the ${String(TCN.train_device).toUpperCase()} device of one workstation and evaluated on its CPU. The trained arm appears only `
+ 'in the comparison of Table XX: the horizon sweeps and the corruption sweep use the oracle, because the checkpoint '
+ 'has a fixed horizon and because those experiments isolate the controller by design.' }]));
body.push(RUNS([{ t: 'Two mechanisms are only partly tested. ', b: true }, { t:
  `An earlier version of this paper stated that the adaptive margin never moved and the safety ceiling never activated. `
+ `Both statements were wrong. The margin moves under every imperfect predictor tested (Sections V-E and V-Q), and the ceiling `
+ `fired in the main comparison, the ablation and the corruption sweep. ${SEC4.E22_SLA_HELPS.length
  ? `Their value is established only narrowly: disabling them raised the violation rate significantly in ${SEC4.E22_SLA_HELPS.length} of the cases in Table XXIII, and in no case in Table IX`
  : `What remains untested is their value: no experiment here shows either one improving the service level significantly${E22_EXISTS ? ', including the ablation under imperfect predictors in Table XXIII' : ''}`}, `
+ 'and the corruption model is symmetric, so it never produces the sustained under-prediction the ceiling exists to catch.' }]));
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
body.push(H1('VIII. Conclusion'));
body.push(P(
  'Three properties of predictive container auto-scaling were measured on a simulated pool, with the predictor held '
+ 'at two extremes so that the control law was the only variable. They differ in how firmly they are established, '
+ 'and the distinction is worth keeping.'));
body.push(P(
  `The forecast horizon must cover the actuation dead time. Across the ${cliffDrop.length} cold-start settings sampling both sides of the boundary, the violation rate fell from `
+ `${f(meanBelow)} % to ${f(meanAbove)} % as the horizon crossed HΔt = τc, and below that boundary the predictive controller performed `
+ `worse than a reactive baseline by up to ${f(Math.abs(worstNeg))} percentage points. Horizon selection is therefore not a tuning preference `
+ 'but a feasibility condition, and it can be checked before deployment from a single measurement of cold-start '
+ 'latency. This is the firmest of the three results on the synthetic families: the transition appeared at every '
+ 'cold-start setting tested and landed where the inequality predicts rather than at a fixed horizon. On the recorded '
+ 'traffic at the scalings used, the reactive baseline never violated, so the transition could not be tested there '
+ '(Section V-Q).'));
body.push(P(
  'The integer problem does not have to be approximated. Dead time confines the tracking-relevant decisions to the '
+ `first H − τ + 1 steps of the horizon and leaves the rest separable, which reduced the search to ${E15.paper_config.m} nested integer `
+ `choices at the configuration used here. The solver built on that reduction reproduced the brute-force optimum in all ${E15.validation.cases} `
+ 'states small enough to enumerate exhaustively, which is a verification rather than a proof but a stronger warrant '
+ 'than the argument it replaces. Completing the horizon greedily instead, a device sometimes called exact on the '
+ `grounds that only the first action is applied, returned an optimal first action in only ${E15.validation.greedy_first_action_optimal} of those `
+ `${E15.validation.cases} states, and at the paper configuration differed from the optimum on ${f(100.0 - E15.paper_config.agree_pct, 2)} % of the plant states drawn, `
+ 'always in the direction of higher cost.'));
body.push(P(
  `Relaxing the integrality of the scaling decision changes behaviour measurably. The relaxed controller disagreed with `
+ `the exact one on ${f(E3R[0].dis, 1)} to ${f(E3R[4].dis, 1)} % of control ticks, issued up to ${(E3R[4].ra / E3R[4].ea).toFixed(1)} times as many scaling actions, and `
+ `degraded from ${f(fr('rounded-oracle', 0.50).sla_mean, 2)} % to ${f(fr('rounded-oracle', 0.85).sla_mean, 2)} % violations as the utilisation target was raised, against `
+ `${f(fr('pact-oracle', 0.50).sla_mean, 2)} % to ${f(fr('pact-oracle', 0.85).sla_mean, 2)} % for the exact controller. Exact solution cost at most `
+ `${f(solveMax, 2)} ms per tick, ${f(solveMax / 50.0, 2)} % of the control interval, so relaxation buys compute that is not scarce `
+ 'at these problem sizes. That is a claim about this configuration rather than a general one: the exact solve is '
+ 'cheap here because the dead time is long relative to the horizon, and a pool with near-instant start-up and a '
+ 'long horizon would not enjoy the same reduction.'));
body.push(P(
  'The results also qualify the approach. At the default operating point the exact controller did not significantly '
+ 'reduce violations relative to either the rounded or the reactive baseline, and it carried more capacity than the '
+ 'rounded controller to achieve the service level it did. Its measured advantages are in actuation stability at the '
+ 'larger pool bounds and in graceful degradation at aggressive operating points. Its adaptive margin and safety ceiling '
+ 'are active, contrary to an earlier version of this paper, but no experiment here establishes that either improves the '
+ `service level${SEC4.E22_SLA_HELPS.length ? ' outside the cases listed in Table XXIII' : ''}, and its dead-time term costs capacity without a measured service-level return.`));
body.push(P(
  'The immediate priorities for further work follow from Section VII: reproducing the horizon–dead-time transition on a '
+ 'real Docker deployment with a measured cold-start interval; repeating the trained-forecaster comparison with several '
+ 'training seeds, several model families and more than one recorded trace; identifying what in the trained forecast '
+ 'damps the rounded controller; and designing a corruption model with sustained bias that would exercise the margin '
+ 'adaptation and the safety ceiling.'));

// ===== References
body.push(H1('References'));
const refs = [
  'T. Van Do et al., "Properties of horizontal pod autoscaling algorithms and application for scaling cloud-native network functions," preprint, 2025.',
  'B. Jeong and Y.-S. Jeong, "ARAScaler: Adaptive resource autoscaling scheme using ETimeMixer for efficient cloud-native computing," IEEE Trans. Services Computing, vol. 18, no. 1, pp. 72–84, 2025, doi: 10.1109/TSC.2024.3522815.',
  'J. W. Fog, J. J. T. Møller, T. M. Jensen, D. Taibi, and M. Albano, "Comparing neural and statistical time-series models for proactive auto-scaling in Kubernetes," in Proc. IEEE Int. Conf. Service-Oriented System Engineering (SOSE), 2025, pp. 151–161, doi: 10.1109/SOSE67019.2025.00022.',
  'E. Incerto, M. Tribastone, and C. Trubiani, "Combined vertical and horizontal autoscaling through model predictive control," in Euro-Par 2018: Parallel Processing, Lecture Notes in Computer Science, vol. 11014, Springer, 2018, pp. 147–159, doi: 10.1007/978-3-319-96983-1_11.',
  'G. Quattrocchi, E. Incerto, R. Pinciroli, C. Trubiani, and L. Baresi, "Autoscaling solutions for cloud applications under dynamic workloads," IEEE Trans. Services Computing, vol. 17, no. 3, pp. 804–820, 2024, doi: 10.1109/TSC.2024.3354062.',
  'D. Zou, W. Lu, Z. Zhu, X. Lu, J. Zhou, X. Wang, K. Liu, K. Wang, R. Sun, and H. Wang, "OptScaler: A collaborative framework for robust autoscaling in the cloud," Proc. VLDB Endowment, vol. 17, no. 12, pp. 4090–4103, 2024, doi: 10.14778/3685800.3685829.',
  'Z. Cai et al., "Deep learning and feedback control based container auto-scaling for cloud native microservices," IEEE Trans. Services Computing, 2025.',
  'F. Meng et al., "CATScaler: A convolution-augmented transformer scaling framework for cloud-native applications," IEEE Trans. Services Computing, 2025.',
  'Z. Fang, H. Ma, G. Chen, and R. Buyya, "HGraphScale: Hierarchical graph learning for autoscaling microservice applications in container-based cloud computing," IEEE Trans. Services Computing, 2025.',
  'B. Kumar, A. Verma, and R. Buyya, "Predictive autoscaling in cloud-native and federated cloud-edge computing environments: A taxonomy and future directions," arXiv:2606.07046, 2026.',
  `${PROV.worldcup98.attribution.replace(/\.$/, '')}.`,
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
    + 'The dashed outline marks the one component that exists in the repository but was not exercised '
    + 'in this evaluation: the Docker actuation backend (no daemon available). The forecaster block is '
    + 'the trained TCN on recorded traffic and the two bracketing predictors everywhere. Bold labels are component names.'),
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

// Split the flat body into sections so that anything too wide for a body
// column (tables of four or more columns, full-width figures) is emitted in
// a one-column continuous section spanning the text block, with its caption
// carried alongside it. Everything else stays in two-column flow.
const TWO_COL = { type: SectionType.CONTINUOUS, page: PAGE,
                  column: { count: 2, space: COL_GAP, equalWidth: true } };
const ONE_COL = { type: SectionType.CONTINUOUS, page: PAGE,
                  column: { count: 1 } };

// Inside a two-column section the columns are balanced, and balancing is
// free to put a figure in one column and its caption in the other. Binding
// each in-column figure to its caption in one unbreakable block prevents it.
function bindFigures(flow) {
  const out = [];
  for (let i = 0; i < flow.length; i += 1) {
    const el = flow[i];
    if (el && el.__img && flow[i + 1] && flow[i + 1].__cap) {
      const group = [el, flow[i + 1]];
      i += 1;
      while (flow[i + 1] && flow[i + 1].__cap) { group.push(flow[i + 1]); i += 1; }
      out.push(B.atomic(group, COL_W));
    } else {
      out.push(el);
    }
  }
  return out;
}

function layout(elements) {
  const out = [];
  let flow = [];
  const flush = () => {
    if (flow.length) {
      out.push({ properties: TWO_COL, footers: { default: foot },
                 children: bindFigures(flow) });
      flow = [];
    }
  };
  for (let i = 0; i < elements.length; i += 1) {
    const el = elements[i];
    if (el && el.__spans) {
      const group = [el];
      // A caption immediately before belongs with this block only if it is
      // not already the caption of a preceding figure -- otherwise a wide
      // table steals the caption of the figure above it.
      const prev = flow[flow.length - 1];
      const prev2 = flow[flow.length - 2];
      if (prev && prev.__cap && !(prev2 && prev2.__img)) {
        group.unshift(flow.pop());
      }
      while (i + 1 < elements.length && elements[i + 1] && elements[i + 1].__cap) {
        group.push(elements[i + 1]);
        i += 1;
      }
      flush();
      out.push({ properties: ONE_COL, footers: { default: foot }, children: group });
    } else {
      flow.push(el);
    }
  }
  flush();
  return out;
}

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
    ...layout(body.slice(0, SPLIT)),
    { properties: ONE_COL, footers: { default: foot }, children: archWide },
    ...layout(body.slice(SPLIT)),
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(ROOT, 'PACT_paper.docx');
  fs.writeFileSync(out, buf);
  console.log('written', out, buf.length, 'bytes');
});
