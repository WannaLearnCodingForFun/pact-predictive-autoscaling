/* Further sections: cliff table, configuration table, positioning table,
 * worked example, and a discussion that synthesises across experiments.
 */
const d = require('docx');
const { Paragraph, TextRun, AlignmentType } = d;
const B = require('./build_paper.js');
const { P, RUNS, H1, H2, H3, CAP, BUL, EQ, sub, sup, img, mkTable,
        E1, E2, E3, E5, E6, E8, E9, E12, MAN, by, f } = B;

const one = (rows, pred) => rows.find(pred);
const fr = (m, u) => E6.find(r => r.method === m && Math.abs(r.u_target - u) < 1e-9);
const TRUTHY = v => v === true || v === 'True' || v === 'true';

// ---------------------------------------------------------- cliff table
const taus = [...new Set(E5.map(r => r.tau_c_s))].sort((a, b) => a - b);
const cliffRows = taus.map(tau => {
  const rows = E5.filter(r => r.tau_c_s === tau).sort((a, b) => a.H - b.H);
  const below = rows.filter(r => !TRUTHY(r.covers_coldstart));
  const above = rows.filter(r => TRUTHY(r.covers_coldstart));
  const mean = a => a.length ? a.reduce((s, r) => s + r.pact_sla, 0) / a.length : NaN;
  const meanR = a => a.length ? a.reduce((s, r) => s + r.reactive_sla, 0) / a.length : NaN;
  return {
    tau, hNeed: Math.ceil(tau / 5),
    firstH: above.length ? above[0].H : NaN,
    below: mean(below), above: mean(above),
    reactive: meanR(rows),
    worstBelow: below.length ? Math.min(...below.map(r => r.improvement_pp)) : NaN,
    bestAbove: above.length ? Math.max(...above.map(r => r.improvement_pp)) : NaN,
  };
});

const cliffTable = [
  mkTable(
    ['τc (s)', 'Ticks of delay\n⌈τc/Δt⌉',
     'First H meeting (6)', 'Violation below\nboundary (%)',
     'Violation above\nboundary (%)', 'Reactive\nviolation (%)',
     'Worst margin\nbelow (pp)', 'Best margin\nabove (pp)'],
    cliffRows.map(r => [
      String(Math.round(r.tau)), String(r.hNeed), String(r.firstH),
      Number.isFinite(r.below) ? f(r.below, 2) : 'n/a',
      f(r.above, 2), f(r.reactive, 2),
      Number.isFinite(r.worstBelow) ? f(r.worstBelow, 2) : 'n/a',
      f(r.bestAbove, 2),
    ]),
    [700, 1200, 1500, 1500, 1500, 1200, 1250, 1250], { left: false, size: 12 }),
  CAP('Table IV. The horizon–dead-time transition, by cold-start setting. Violation columns '
  + 'average over the horizons on each side of the boundary HΔt = τc. The margin columns give the '
  + 'difference from the reactive baseline, where a negative value means the predictive controller '
  + 'was worse. At τc = 5 s the shortest horizon tested already satisfies (6), so no value falls '
  + 'below the boundary. Synthetic traces, 3 families × 3 seeds per cell.'),
  P('Reading the table by row makes the mechanism concrete. At τc = 45 s the plant needs '
  + `${cliffRows[3].hNeed} ticks of lookahead; the sweep tested H = 8 and H = 12 on either side, and the violation rate `
  + `falls from ${f(cliffRows[3].below, 2)} % to ${f(cliffRows[3].above, 2)} % between them. The reactive baseline on the same traces sits at `
  + `${f(cliffRows[3].reactive, 2)} %, so an under-configured predictive controller is ${f(Math.abs(cliffRows[3].worstBelow), 2)} points worse than not `
  + `predicting at all, while a correctly configured one is ${f(cliffRows[3].bestAbove, 2)} points better. The same pattern holds at `
  + 'every delay tested, and the boundary moves with τc exactly as (6) requires rather than at some '
  + 'fixed horizon that would indicate a different cause.'),
];

// ---------------------------------------------------------- worked example
const workedExample = [
  H3('A worked example'),
  P('A concrete instance makes the condition tangible. Suppose a pool of eight replicas is serving '
  + '260 requests per second at a 65 % target, demand doubles over the next thirty seconds, and the '
  + 'container image takes 45 s to become ready. The requirement will rise to roughly sixteen '
  + 'replicas. A reactive controller observes the rise only as utilisation crosses its target, issues '
  + 'the scale-up at that moment, and the new replicas begin serving 45 s later, by which point the '
  + 'pool has been under-provisioned for the entire rise and part of the plateau.'),
  P('A predictive controller with a one-step horizon is no better. It forecasts five seconds ahead, '
  + 'which tells it nothing about a transition that completes in thirty, and its scale-up lands at '
  + 'essentially the same moment as the reactive one, now with forecast error added. Only when the '
  + 'horizon reaches nine ticks does the forecast span the actuation delay, and only then can the '
  + 'controller issue at t the action whose effect arrives at t + 45 s, when the demand does. The '
  + 'measurements in Section V-A are this example repeated across six delays and ten horizons, and '
  + 'they show the transition occurring where this reasoning predicts.'),
];

// ---------------------------------------------------------- configuration table
const configTable = [
  H2('F. Configuration and Reproducibility'),
  P('Table III lists every parameter that affects the reported results. Values marked as swept are '
  + 'varied in the experiment indicated; all others are held fixed throughout. The repository '
  + 'reproduces every result file and figure from this configuration with the command sequence given '
  + 'at the end of this subsection.'),
  mkTable(
    ['Symbol', 'Parameter', 'Value', 'Set by'],
    [
      ['Δt', 'Control interval', '5 s', 'Fixed'],
      ['H', 'Forecast horizon', '12 ticks (60 s)', 'Swept in E5, E9'],
      ['τc', 'Cold-start latency', '45 s', 'Swept in E5, E12'],
      ['μ', 'Per-replica service rate', '40 req/s', 'Fixed'],
      ['u*', 'Target utilisation', '0.65', 'Swept in E6'],
      ['r*', 'Target memory fraction', '0.75', 'Fixed'],
      ['ℓmax', 'Latency objective', '200 ms', 'Fixed'],
      ['cₐ², cₛ²', 'Squared CVs of arrival, service', '1.0, 1.0', 'Fixed'],
      ['Nmin, Nmax', 'Pool bounds', '1, 16', 'Nmax swept in E3'],
      ['Δmax', 'Per-tick rate limit', '4 replicas', 'Swept in E8, E9'],
      ['q⁺ / q⁻', 'Tracking asymmetry', '8 / 1', 'Swept in E8'],
      ['r', 'Actuation weight', '0.15', 'Fixed'],
      ['s', 'Churn weight', '0.60', 'Swept in E8'],
      ['θ', 'Deadband', '1 replica', 'Swept in E8'],
      ['T_cool', 'Cooldown', '60 s', 'Swept in E8'],
      ['u_emergency', 'Safety-ceiling threshold', '0.92', 'Fixed'],
      ['κγ, η', 'Margin gain, error smoothing', '0.02, 0.20', 'Fixed'],
      ['γmin, γmax', 'Margin bounds', '0.00, 0.50', 'Fixed'],
      ['—', 'Ticks per run', String(MAN.n_ticks_per_run), 'Fixed'],
      ['—', 'Seeds per configuration', `${MAN.n_seeds} (E1, E4); 3–8 elsewhere`, 'Per experiment'],
      ['—', 'Latency censoring ceiling', '5000 ms', 'Fixed'],
    ],
    [1200, 3000, 2400, 2000], { left: true, size: 13 }),
  CAP('Table III. Complete parameter set. Values are those used unless the experiment named in the '
  + 'final column varies them.'),
  RUNS([
    { t: 'Reproduction. ', b: true },
    { t: 'From a clean checkout: python3 experiments/run_all.py writes the primary result files; '
       + 'python3 experiments/run_more.py writes the sensitivity, timing and breakdown files; '
       + 'python3 experiments/make_figures.py and make_figures2.py regenerate every plot; '
       + 'node experiments/paper.js rebuilds this document, reading each reported value from the '
       + 'result files at build time. Total runtime for the experiment stage is approximately six '
       + 'minutes on the platform recorded in results/manifest.json.' },
  ]),
];

// ---------------------------------------------------------- positioning table
const positioning = [
  H2('P. Positioning Against Prior Work'),
  P('Table XVIII places the present work against the reviewed systems on the three axes the '
  + 'experiments address. The comparison is of mechanism, not of reported numbers: none of the '
  + 'systems in the table was reimplemented, their published results were obtained on different '
  + 'plants and workloads, and no numerical claim is made against them.'),
  mkTable(
    ['Property', 'Threshold [1]', 'Forecast-driven [2,3,8]',
     'Optimisation [4,5,6]', 'Hybrid [7,11]', 'This work'],
    [
      ['Decision variable is integer', 'yes', 'varies', 'no, rounded', 'varies', 'yes, solved exactly'],
      ['Actuation delay modelled', 'no', 'no', 'no', 'partially', 'yes, as plant dead time'],
      ['Horizon tied to delay', 'n/a', 'no', 'no', 'no', 'yes, Eq. (6), measured'],
      ['Decision churn penalised', 'via stabilisation window', 'no', 'no', 'no', 'yes, in the objective'],
      ['Churn reported as a metric', 'no', 'no', 'no', 'no', 'yes'],
      ['Hard pool and rate constraints', 'yes', 'varies', 'yes', 'varies', 'yes'],
      ['Conservatism mechanism', 'static threshold', 'none', 'chance constraint', 'feedback term', 'clipped-integral margin'],
      ['Evaluated on real hardware', 'yes', 'varies', 'yes', 'yes', 'no, simulated plant'],
    ],
    [2500, 1700, 2100, 1800, 1500, 2100], { left: true, size: 12 }),
  CAP('Table XVIII. Mechanism comparison. Entries describe design choices reported by the cited '
  + 'work, not relative performance. The final row records the principal weakness of the present '
  + 'evaluation.'),
  P('Three columns separate this work from the rest — exact integrality, a horizon tied to '
  + 'measured delay, and churn treated as both a penalty and a reported metric — and those are '
  + 'precisely the three the experiments quantify. The last row runs the other way: every other '
  + 'system in the comparison was evaluated on real infrastructure, and this one was not.'),
];

// ---------------------------------------------------------- discussion
const po = by(E1, 'method', 'pact-oracle');
const ro = by(E1, 'method', 'rounded-oracle');
const re = by(E1, 'method', 'reactive');
const sigSla = E2.find(r => r.method === 'rounded-oracle' && r.metric === 'sla');
const sigAct = E2.find(r => r.method === 'rounded-oracle' && r.metric === 'act');
const cd0 = one(E8, r => r.parameter === 'cooldown_s' && r.value === 0.0);
const cd60 = one(E8, r => r.parameter === 'cooldown_s' && r.value === 60.0);

const discussion = [
  H1('VI. Discussion'),
  H2('A. What the Measurements Support'),
  P('Three claims survive the full set of experiments, and it is worth separating them from the '
  + 'claims the design anticipated but the data did not sustain.'),
  P('The horizon condition is the strongest result and the one with the clearest practical reading. '
  + 'It holds at every cold-start setting tested, the transition lands where (6) predicts rather than '
  + 'at a fixed horizon, and the sign of the predictive controller’s advantage reverses across it. '
  + 'It also requires no tuning to exploit: the operator measures one latency and computes one '
  + 'inequality. Section V-J strengthens it by showing that once the horizon is adequate, the '
  + 'controller is nearly insensitive to the delay itself over a factor of twelve.'),
  P('The integrality result is narrower but holds under both forecaster brackets. Exact and rounded '
  + 'controllers differ on a large and growing fraction of decisions, the difference materialises as '
  + 'churn rather than as violations at moderate operating points, and it materialises as a service '
  + 'collapse at aggressive ones. The statistical picture in Table VIII is consistent with this: the '
  + `churn difference is unambiguous (Cliff’s δ = ${f(sigAct.cliffs_delta, 2)}, p < 0.001) while the violation difference at the `
  + `default operating point is not significant after correction (p = ${f(sigSla.p_holm, 3)}).`),
  P('The third claim is about stability mechanisms, and it inverts the design’s assumption. The '
  + 'cooldown, which was included as a conventional guard rail, turned out to dominate: removing it '
  + `raises churn from ${f(cd60.act_mean, 1)} to ${f(cd0.act_mean, 1)} actions per run. The churn penalty in the objective matters too, `
  + 'and improves service level and cost simultaneously rather than trading between them, but it '
  + 'saturates quickly. The deadband and the cost asymmetry, which the design treated as primary '
  + 'controls, were close to inert.'),
  H2('B. What the Measurements Do Not Support'),
  P('Four anticipated effects were absent, and reporting them is as informative as reporting the '
  + 'successes because each identifies either a redundant mechanism or an untested one.'),
  P('The explicit dead-time term in the plant model contributed nothing detectable when the horizon '
  + 'was already adequate. The optimiser compensates for delay implicitly through the horizon, and '
  + 'the delay term is a formal convenience rather than a separate source of benefit. The paper '
  + 'therefore attributes dead-time tolerance to the predictive formulation as a whole.'),
  P('The adaptive margin never moved. Under a perfect predictor the tracking error is negligible, and '
  + 'under a memoryless one it is large but zero-mean, so the clipped integral has nothing to '
  + 'integrate in either case. The mechanism is untested rather than unnecessary, and testing it '
  + 'requires a predictor whose error has sustained sign, which is exactly what a drifting learned '
  + 'model would produce and neither bracket does.'),
  P('The safety ceiling never activated at any corruption level tested. The same caveat applies: '
  + 'symmetric multiplicative noise does not produce the sustained under-prediction the ceiling '
  + 'exists to catch. Section V-F shows noise actually improved service level by inflating capacity '
  + 'through the asymmetric cost, which is a property of the objective rather than evidence of '
  + 'robustness.'),
  P('Finally, the proposed controller did not dominate. It carries more capacity than the rounded '
  + `controller at matched settings (${f(po.over_mean, 1)} % against ${f(ro.over_mean, 1)} %) and issues more scaling actions than the `
  + `reactive baseline (${f(po.act_mean, 1)} against ${f(re.act_mean, 1)}). An operator whose workload resembles the slow families in `
  + 'Table XIV, and who can tolerate the rounded controller’s violation rate, has no reason to prefer '
  + 'exact solution on these results.'),
  H2('C. When Each Mechanism Is Worth Its Cost'),
  P('Synthesising across the experiments gives a decision rule that is narrower than a general '
  + 'recommendation but better supported than one. Exact integer solution is worth adopting when the '
  + 'pool is small relative to demand swings, when the target utilisation is aggressive, or when the '
  + 'workload has fast transitions — the three conditions under which Table V, Table VI and Table XIV '
  + 'respectively show the largest separation. It is not worth adopting on slowly varying traffic at '
  + 'conservative targets, where the measurements show no advantage and a capacity penalty.'),
  P('Predictive scaling in general is worth adopting when the actuation delay is long relative to the '
  + 'control interval, which Section V-J quantifies, and only when the horizon can be configured to '
  + 'cover it. Where the delay is short — at τc = 5 s the reactive baseline violates on '
  + `${f(cliffRows[0].reactive, 2)} % of ticks against ${f(cliffRows[0].above, 2)} % predictive — the additional machinery buys little.`),
];

module.exports = { cliffTable, workedExample, configTable, positioning, discussion };

// ---------------------------------------------------------- predictor quality
const E13 = B.csv('e13_predictor_quality.csv');
const pq = (m, w) => E13.find(r => r.method === m && Math.abs(r.oracle_weight - w) < 1e-9);
const pqBest = E13.filter(r => r.method === 'pact')
  .reduce((a, r) => (r.sla_mean < a.sla_mean ? r : a));
const pqBestR = E13.filter(r => r.method === 'rounded')
  .reduce((a, r) => (r.sla_mean < a.sla_mean ? r : a));

const predictorQuality = [
  H2('M. Response to Predictor Quality'),
  P('The bracketing protocol of Section IV-C bounds the predictor’s contribution but leaves the '
  + 'interior unmeasured, and a real forecaster lives in that interior. To fill it, a blended '
  + 'predictor was constructed that returns a convex combination of the two brackets, with weight w '
  + 'on perfect foresight and 1 − w on persistence. Sweeping w traces the controller’s response to '
  + 'predictor quality without requiring any particular model to be trained. Table XVI reports the '
  + 'sweep and Fig. 17 shows it.'),
  mkTable(
    ['w', 'Exact MPC\nviolation (%)', 'Exact MPC\nover-prov. (%)', 'Exact MPC\nactions',
     'Rounded\nviolation (%)', 'Rounded\nover-prov. (%)', 'Rounded\nactions'],
    [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0].map(w => [
      f(w, 1),
      f(pq('pact', w).sla_mean, 2), f(pq('pact', w).over_mean, 1), f(pq('pact', w).act_mean, 1),
      f(pq('rounded', w).sla_mean, 2), f(pq('rounded', w).over_mean, 1), f(pq('rounded', w).act_mean, 1),
    ]),
    [700, 1400, 1500, 1200, 1400, 1500, 1200], { left: false, size: 13 }),
  CAP(`Table XVI. Response to predictor quality. w = 0 is the persistence bracket and w = 1 the `
  + `perfect-foresight bracket. Synthetic traces, 5 families × 6 seeds = ${pq('pact', 0.0).n_runs} runs per cell.`),
  P('The response is not monotone, and that is the substantive finding. For the integer-exact '
  + `controller the violation rate falls from ${f(pq('pact', 0.0).sla_mean, 2)} % at pure persistence to a minimum of `
  + `${f(pqBest.sla_mean, 2)} % at w = ${pqBest.oracle_weight}, then rises again to ${f(pq('pact', 1.0).sla_mean, 2)} % at perfect foresight. The rounded controller shows `
  + `the same shape with its minimum at w = ${pqBestR.oracle_weight}. A perfectly accurate forecast is therefore not the best `
  + 'forecast for either controller on these workloads.'),
  P('The mechanism is visible in the churn panel of Fig. 17. A perfect forecast reproduces every '
  + 'fluctuation in demand, including the noise, and the controller tracks it; actions rise from '
  + `${f(pqBest.act_mean, 1)} at the optimum to ${f(pq('pact', 1.0).act_mean, 1)} at w = 1. Blending in a persistence component low-pass filters the `
  + 'forecast, and the resulting slight lag behaves like a small damping term. The controller acts '
  + 'less often, holds a steadier pool, and ends up violating less. The effect is stronger for the '
  + `rounded controller, whose action count more than doubles from ${f(pqBestR.act_mean, 1)} at its optimum to `
  + `${f(pq('rounded', 1.0).act_mean, 1)} at perfect foresight, because its fractional oscillation compounds with the unfiltered signal.`),
  P('Two consequences follow for how such systems should be built and evaluated. First, optimising a '
  + 'forecaster for accuracy alone optimises the wrong objective, which is the methodological point '
  + 'the paper makes in Section I and here measures directly: an accuracy improvement that sharpens '
  + 'the predicted signal can degrade the closed loop. Second, the bracketing protocol is '
  + 'conservative in an unexpected direction. Because the interior of the range outperforms both '
  + 'endpoints, the reported bracket does not contain the best achievable result, and a trained '
  + 'forecaster of intermediate sharpness could plausibly perform better than either bracket rather '
  + 'than between them.'),
  P('The result should not be over-read. It was obtained on synthetic traces whose noise is additive '
  + 'and Gaussian, and the damping benefit depends on that noise being uninformative. On a workload '
  + 'whose high-frequency content carries real demand, filtering it away would cost service level '
  + 'rather than save it. What the experiment establishes is that the relationship between forecast '
  + 'accuracy and control performance is not monotone, not the location of the optimum in general.'),
  img('fig_predictor_quality.png', 6.9),
  CAP('Fig. 17. Controller response to predictor quality. The violation-rate minimum for both '
  + 'controllers lies strictly inside the bracket rather than at perfect foresight. Synthetic traces.'),
];

module.exports.predictorQuality = predictorQuality;
