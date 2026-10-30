/* Expanded sections: literature table, algorithms, new results.
 * Exported as arrays of docx elements; paper.js splices them in.
 */
const d = require('docx');
const { Paragraph, TextRun, AlignmentType } = d;
const B = require('./build_paper.js');
const { P, RUNS, H2, H3, CAP, BUL, img, mkTable, ALGO,
        E8, E9, E10, E11, E12, E15, E15T, E5X, E8C, MAN, f } = B;

const sel = (rows, k, v) => rows.filter(r => r[k] === v);
const one = (rows, pred) => rows.find(pred);

// ============================================================ literature table
const LIT = [
  ['[1]', 'Threshold', 'HPA control law, analytic characterisation',
   'Session-based CNF traffic', 'Finite distinct tolerance intervals; bounds on reactive scaling',
   'Characterises the baseline only; no predictive mechanism'],
  ['[2]', 'Forecast-driven', 'ETimeMixer with convolution; burst/non-burst state classification',
   'Seven public cloud workload traces', 'Reports ≈ 70 % utilisation with fewer scaling updates',
   'Scaling rule is a state-dependent heuristic, not an optimisation'],
  ['[3]', 'Forecast-driven', 'Controlled comparison of statistical and neural forecasters',
   'Kubernetes workload traces', 'Neural advantage is workload-dependent, not universal',
   'Comparison study; contributes no scaling mechanism'],
  ['[4]', 'Optimisation', 'MPC over combined vertical and horizontal scaling',
   'Containerised web applications', 'First MPC formulation for container elasticity',
   'Real-valued decisions; no dead-time term; no churn metric'],
  ['[5]', 'Optimisation', 'Queueing-network state estimation, 1 s control interval',
   'Two applications, seven workloads', 'Near-zero SLA violations at sub-millisecond decision cost',
   'Reactive state estimation; no lookahead across actuation delay'],
  ['[6]', 'Optimisation', 'Proactive forecaster + reactive estimator in one chance-constrained MPC',
   'Alipay production traces, three clusters', 'Reports > 36 % fewer SLO violations than a hybrid baseline',
   'Real-valued decisions, rounded for use; no integrality constraint; no churn penalty; 30 min intervals'],
  ['[7]', 'Hybrid learning + control', 'Deep forecaster supplying feedforward, feedback law correcting residual',
   'Containerised microservice benchmarks', 'Feedback term stabilises the loop when the predictor degrades',
   'Controller gains tuned per application; no hard constraints'],
  ['[8]', 'Forecast-driven', 'Convolution-augmented Transformer, local and global workload structure',
   'Cloud-native and serverless traces', 'Captures short bursts and long periodicity jointly',
   'Model footprint impractical for sidecar deployment'],
  ['[9]', 'Graph / RL', 'Hierarchical graph neural network over container dependencies',
   'Real-world microservice traces', 'Captures inter-service coupling in the scaling decision',
   'No safety guarantee during exploration; sample-inefficient'],
  ['[10]', 'Survey', 'Four-dimensional taxonomy over triggers, targets, models, metrics',
   'Systematic review', 'Maps the design space; identifies drift-awareness gap',
   'No empirical evaluation of the proposed indices'],
];

const litSection = [
  H2('F. Comparative Summary'),
  P('Table I consolidates the reviewed work. The grouping is by the mechanism each study uses to '
  + 'decide the replica count, because that is the axis along which the present contribution is '
  + 'positioned. Two columns deserve attention. The methodology column records whether the decision '
  + 'variable is discrete; across the optimisation group it is not. The limitation column records '
  + 'whether actuation delay and decision churn are treated as first-class quantities; across all '
  + 'groups they are not.'),
  mkTable(
    ['Ref.', 'Mechanism', 'Methodology', 'Dataset / benchmark', 'Reported outcome', 'Limitation'],
    LIT, [600, 1500, 2650, 1900, 2350, 2500], { left: true, size: 13 }),
  CAP('Table I. Reviewed work grouped by decision mechanism. Outcomes are as reported by the '
  + 'original authors and were not reproduced here; the present work reproduces none of these '
  + 'systems and makes no numerical comparison against their published figures.'),
  P('Nowhere in the table is the replica count treated as an integer decision variable. The '
  + 'optimisation group, the closest antecedent of this work, relaxes integrality and rounds the '
  + 'solution, and none of those studies reports what the rounding costs. Actuation delay is '
  + 'handled explicitly in a single entry, and there at a granularity two orders of magnitude '
  + 'slower than container start-up. Decision churn is reported nowhere at all, although every '
  + 'scaling action consumes a container start or stop and redistributes live connections. Those '
  + 'three omissions are what Section V measures.'),
];

// ============================================================ algorithms
const algos = [
  H2('E. Algorithms'),
  P('Algorithm 1 states the per-tick control loop. Lines 2 to 4 realise the observation and capacity '
  + 'mapping of Section III-A, line 6 the safety ceiling, line 9 the integer optimisation, and lines '
  + '11 to 14 the feasibility projections and gates in the order given in Section III-C. The order is '
  + 'load-bearing: projections precede gates, so an unusual forecast cannot defeat the pool bounds or '
  + 'the rate limit by routing around them.'),
  ALGO('Algorithm 1: PACT control tick', [
    ' 1  input  N(t), pending, u_prev, γ(t), config',
    ' 2  obs   ← observe_pool()                      // Eq. (1)',
    ' 3  Ŷ     ← forecast(obs, H)                     // predictor',
    ' 4  Ñ[1..H] ← capacity_map(Ŷ, N(t), γ)          // Eq. (2)-(4)',
    ' 5',
    ' 6  if obs.u > u_emergency then                  // ceiling',
    ' 7      return immediate_scale_up(N(t), obs.u)',
    ' 8',
    ' 9  u* ← argmin J(u) over u ∈ [−Δmax, Δmax] ∩ ℤ  // Eq. (7)-(9)',
    '10',
    '11  u ← clip(u*, −Δmax, Δmax)                    // rate limit',
    '12  u ← clip(N(t)+u, Nmin, Nmax) − N(t)           // pool bounds',
    '13  if |u| < θ then u ← 0                         // deadband',
    '14  if u < 0 and t − t_up < T_cool then u ← 0     // cooldown',
    '15',
    '16  actuate(N(t) + u)',
    '17  γ(t+1) ← margin_update(γ(t), e(t))            // Eq. (11)',
    '18  return u',
  ], 'Algorithm 1. One control tick. Gates (13–14) can only suppress an action, never create one.'),
  P('Algorithm 2 states the integer solve of line 9. Solving it exactly looks expensive: the '
  + 'decision space has (2Δmax + 1)ᴴ points, which is 9¹² at the configuration used here. Two '
  + 'properties of the dead-time plant reduce it to something that can be enumerated.'),
  P('The first is truncation. Because u(t+j) does not appear in N until tick t + j + τ, it can '
  + 'enter a tracking term (N(t+k+1) − Ñ(t+1+k))² only when j + τ ≤ k + 1. Over k = 0 … H − 1 that '
  + 'confines the tracking-relevant actions to j ≤ H − τ, so only m = max(1, H − τ + 1) of the H '
  + 'decision variables can move the tracking cost at all. Saturation of N at the pool bounds does '
  + 'not weaken the argument, because saturation depends only on what has already landed. At τ = 9 '
  + 'and H = 12 this gives m = 4, an enumeration over 9⁴ = 6561 sequences rather than 9¹².'),
  P('The second is separability of the remainder. Actions u(t+m) … u(t+H−1) land after the horizon '
  + 'ends, so they contribute only the decoupled chain Σ[r u² + s(u − u_prev)²] and their own '
  + 'admissibility. That is a scalar shortest-path problem in (committed, u_prev), minimised '
  + 'exactly by backward recursion. The remainder is not identically zero: with r = 0.15, s = 0.60 '
  + 'and u(t+m−1) = 4, holding the tail at zero costs 9.60 against 4.35 for a ramp-down. A '
  + 'completion that zeroes the tail therefore overstates the cost of every plan ending on a large '
  + 'action, and because the overstatement depends on the plan it does not cancel in the argmin.'),
  P('Memoising the recursion on the full controllable state (k, N, pending, u_prev) keeps the '
  + 'procedure tractable at the opposite extreme. When τ is small the pending queue is short, so '
  + 'distinct action prefixes collapse onto the same state and the recursion is polynomial; when τ '
  + 'is large the queue is long but m is small and the tree is shallow. The two regimes are '
  + 'complementary, which is why a single procedure covers the whole grid swept in Section V-A.'),
  ALGO('Algorithm 2: Globally optimal integer first action', [
    ' 1  input  N, Ñ[1..H], pending, u_prev, config',
    ' 2  m ← max(1, H − τ + 1)                        // truncation',
    ' 3  memo ← { }',
    ' 4',
    ' 5  function V(k, n, p, prev):                   // cost-to-go',
    ' 6      if k = m then',
    ' 7          return tail_track(n, p, Ñ) + chain(n, p, prev)',
    ' 8      if (k, n, p, prev) ∈ memo then return memo[·]',
    ' 9      v ← ∞;  cm ← committed(n, p)',
    '10      for a ∈ {−Δmax, …, +Δmax} do',
    '11          if cm + a ∉ [Nmin, Nmax] then continue',
    '12          n′, p′ ← plant(n, p, a, τ)           // Eq. (5)',
    '13          q ← (n′ < Ñ[k+1]) ? q⁺ : q⁻          // Eq. (8)',
    '14          c ← q(n′ − Ñ[k+1])² + r a² + s(a − prev)²',
    '15          v ← min(v, c + V(k+1, n′, p′, a))',
    '16      memo[(k, n, p, prev)] ← v;  return v',
    '17',
    '18  best ← ∞;  u_best ← 0;  cm ← committed(N, pending)',
    '19  for u ∈ {−Δmax, …, 0, …, +Δmax} do',
    '20      if cm + u ∉ [Nmin, Nmax] then continue',
    '21      n′, p′ ← plant(N, pending, u, τ)',
    '22      q ← (n′ < Ñ[1]) ? q⁺ : q⁻',
    '23      c ← q(n′ − Ñ[1])² + r u² + s(u − u_prev)²',
    '24      if c + V(1, n′, p′, u) < best then',
    '25          best ← c + V(1, n′, p′, u);  u_best ← u',
    '26  return u_best',
  ], 'Algorithm 2. Exact integer solve. tail_track evaluates the tracking terms of stages '
   + 'm … H−1, which the truncation argument fixes independently of the remaining actions; '
   + 'chain is the backward recursion over the decoupled actuation and churn terms of those '
   + 'stages. Both are exact, so the returned first action is the global optimum.'),
  P('The cost is O((2Δmax + 1)ᵐ · H) when the dead time is long and polynomial in H when it is '
  + 'short, with no iteration to convergence and therefore no data-dependent worst case. '
  + 'Section V-H reports the measured time.'),
  P('Correctness was checked against exhaustive enumeration rather than argued. Across ten '
  + 'configurations small enough to enumerate the full (2Δmax + 1)ᴴ decision space, the first '
  + `action returned by Algorithm 2 was compared with brute force on ${E15.validation.cases} randomly drawn plant `
  + `states; it attained the brute-force optimum in ${E15.validation.exact_attains_optimum} of them. The same comparison prices `
  + 'the alternative of completing the horizon greedily: a solver that enumerates first actions and '
  + 'then tracks demand myopically returned an optimal first action in only '
  + `${E15.validation.greedy_first_action_optimal} of those ${E15.validation.cases} cases. Section V-I quantifies the gap at the configuration used `
  + 'throughout this paper.'),
];

// ============================================================ V-F control sensitivity
function pick(param, v) { return one(E8, r => r.parameter === param && Math.abs(r.value - v) < 1e-9); }

const s0 = pick('churn_weight_s', 0.0);
const s1 = pick('churn_weight_s', 1.0);
const s2 = pick('churn_weight_s', 2.0);
const cd0 = pick('cooldown_s', 0.0);
const cd60 = pick('cooldown_s', 60.0);
const cd120 = pick('cooldown_s', 120.0);
const th1 = pick('deadband_theta', 1);
const th2 = pick('deadband_theta', 2);
const th3 = pick('deadband_theta', 3);
const q1 = pick('asymmetry_qup_over_qdown', 1.0);
const q20 = pick('asymmetry_qup_over_qdown', 20.0);
const dm1 = pick('rate_limit_delta_max', 1);
const dm8 = pick('rate_limit_delta_max', 8);

const sensitivity = [
  H2('G. Control-Parameter Sensitivity'),
  P('The controller exposes five parameters that shape the trade-off between service level and '
  + 'actuation stability. Each was swept independently over the five workload families with eight '
  + 'seeds, holding the others at their defaults. Table X reports the sweeps and Fig. 10 shows '
  + 'them. The results reorder the components by importance relative to what the design assumed.'),
  mkTable(
    ['Parameter', 'Value', 'Violation (%)', 'Over-prov. (%)', 'Actions', 'Oscillations', 'Cost /1k'],
    [
      ['Churn weight s', '0.00', f(s0.sla_mean, 2), f(s0.over_mean, 1), f(s0.act_mean, 1), f(s0.osc_mean, 1), f(s0.cost_mean, 1)],
      ['', '1.00', f(s1.sla_mean, 2), f(s1.over_mean, 1), f(s1.act_mean, 1), f(s1.osc_mean, 1), f(s1.cost_mean, 1)],
      ['', '2.00', f(s2.sla_mean, 2), f(s2.over_mean, 1), f(s2.act_mean, 1), f(s2.osc_mean, 1), f(s2.cost_mean, 1)],
      ['Cooldown T_cool (s)', '0', f(cd0.sla_mean, 2), f(cd0.over_mean, 1), f(cd0.act_mean, 1), f(cd0.osc_mean, 1), f(cd0.cost_mean, 1)],
      ['', '60', f(cd60.sla_mean, 2), f(cd60.over_mean, 1), f(cd60.act_mean, 1), f(cd60.osc_mean, 1), f(cd60.cost_mean, 1)],
      ['', '120', f(cd120.sla_mean, 2), f(cd120.over_mean, 1), f(cd120.act_mean, 1), f(cd120.osc_mean, 1), f(cd120.cost_mean, 1)],
      ['Deadband θ', '1', f(th1.sla_mean, 2), f(th1.over_mean, 1), f(th1.act_mean, 1), f(th1.osc_mean, 1), f(th1.cost_mean, 1)],
      ['', '2', f(th2.sla_mean, 2), f(th2.over_mean, 1), f(th2.act_mean, 1), f(th2.osc_mean, 1), f(th2.cost_mean, 1)],
      ['', '3', f(th3.sla_mean, 2), f(th3.over_mean, 1), f(th3.act_mean, 1), f(th3.osc_mean, 1), f(th3.cost_mean, 1)],
      ['Asymmetry q⁺/q⁻', '1', f(q1.sla_mean, 2), f(q1.over_mean, 1), f(q1.act_mean, 1), f(q1.osc_mean, 1), f(q1.cost_mean, 1)],
      ['', '20', f(q20.sla_mean, 2), f(q20.over_mean, 1), f(q20.act_mean, 1), f(q20.osc_mean, 1), f(q20.cost_mean, 1)],
      ['Rate limit Δmax', '1', f(dm1.sla_mean, 2), f(dm1.over_mean, 1), f(dm1.act_mean, 1), f(dm1.osc_mean, 1), f(dm1.cost_mean, 1)],
      ['', '8', f(dm8.sla_mean, 2), f(dm8.over_mean, 1), f(dm8.act_mean, 1), f(dm8.osc_mean, 1), f(dm8.cost_mean, 1)],
    ],
    [1800, 900, 1300, 1400, 1000, 1300, 1000], { left: false, size: 13 }),
  CAP(`Table X. Control-parameter sweeps. Synthetic traces, 5 families × 8 seeds = ${s0.n_runs} runs `
  + 'per row; the remaining parameters are held at their defaults. Oscillations counts sign reversals '
  + 'in the replica delta.'),
  P('The churn penalty is not a pure trade-off. Raising s from zero to unity reduces violations from '
  + `${f(s0.sla_mean, 2)} % to ${f(s1.sla_mean, 2)} %, over-provisioning from ${f(s0.over_mean, 1)} % to `
  + `${f(s1.over_mean, 1)} %, and oscillations from ${f(s0.osc_mean, 1)} to ${f(s1.osc_mean, 1)}, improving all three at once. `
  + 'Beyond unity the curve is flat. The mechanism is that penalising change suppresses the '
  + 'scale-down-then-scale-up pairs that would otherwise remove capacity immediately before it is '
  + 'needed again, so the controller holds a steadier pool that happens also to be a smaller one. '
  + 'This was not an anticipated result; the penalty was introduced to reduce churn alone.'),
  P('The cooldown dominates stability, and by a wider margin than the churn penalty does. Removing it '
  + `entirely raises the action count from ${f(cd60.act_mean, 1)} to ${f(cd0.act_mean, 1)} per run and oscillations from `
  + `${f(cd60.osc_mean, 1)} to ${f(cd0.osc_mean, 1)}, while violations rise from ${f(cd60.sla_mean, 2)} % to ${f(cd0.sla_mean, 2)} %. `
  + `Extending it to ${cd120.value} s continues to help on both axes, reaching ${f(cd120.act_mean, 1)} actions at `
  + `${f(cd120.sla_mean, 2)} % violations for ${f(cd120.over_mean, 1)} % over-provisioning. Fig. 11 plots this sweep in the `
  + 'violation-versus-churn plane; the curve is monotone, so on these workloads the cooldown is close '
  + 'to free. The practical implication is that an operator tuning this controller should set the '
  + 'cooldown first and the churn weight second.'),
  P(`The deadband is inert at its default. Values of θ = ${th1.value} and θ = 0 give identical results, because an `
  + 'integer action whose magnitude is below one is already zero; the deadband only becomes active at '
  + `θ = ${th2.value}, where it cuts actions from ${f(th1.act_mean, 1)} to ${f(th2.act_mean, 1)} at a cost of `
  + `${f(th2.sla_mean - th1.sla_mean, 2)} percentage points of service level, and at θ = ${th3.value} it becomes harmful, reaching `
  + `${f(th3.sla_mean, 2)} % violations. The asymmetric tracking weight is similarly weak: raising q⁺/q⁻ from `
  + `${q1.value} to ${q20.value} moves violations only from ${f(q1.sla_mean, 2)} % to ${f(q20.sla_mean, 2)} %. Both findings are reported `
  + 'because they contradict the design intent, in which the deadband and the cost asymmetry were '
  + 'expected to be primary controls.'),
  img('fig_sensitivity.png', 6.9),
  CAP('Fig. 10. Control-parameter sweeps. Blue circles on the left axis are the violation rate; '
  + 'orange triangles on the right axis are the oscillation count. Synthetic traces.'),
  img('fig_cooldown.png', 3.35),
  CAP('Fig. 11. The cooldown sweep in the violation-versus-churn plane. Points are labelled with '
  + 'the cooldown that produced them. Synthetic traces.'),
];

// ============================================================ V-G solve time
const e9max = E9.reduce((a, r) => Math.max(a, r.max_solve_ms), 0);
const e9small = one(E9, r => r.delta_max === 1 && r.H === 4);
const e9big = one(E9, r => r.delta_max === 16 && r.H === 30);
const e9def = one(E9, r => r.delta_max === 4 && r.H === 12);

const solvetime = [
  H2('H. Computational Cost of Exact Solution'),
  P('If exact integer solution were expensive, relaxation would be justified regardless of the '
  + 'accuracy argument. Algorithm 2 was timed across the action-set size and the horizon, the two '
  + 'parameters that determine its work, with the dead time held at its default. The sweep answers '
  + 'two questions at once: what the exact solve costs where it is available, and where it stops '
  + 'being available at all. Table XI reports the extremes and Fig. 12 the full grid. Cells that '
  + 'exceeded the node budget were solved by the greedy completion instead and are marked as such; '
  + 'their timings describe that solver, not this one.'),
  mkTable(
    ['Δmax', 'Candidates', 'H', 'Solver', 'Mean solve (ms)', 'Max (ms)',
     'Share of 5 s budget (%)'],
    E9.filter(r => [1, 4, 16].includes(r.delta_max) && [4, 12, 30].includes(r.H))
      .map(r => [String(r.delta_max), String(r.candidates), String(r.H),
                 (r.solver === undefined || r.solver === '')
                   ? '—'
                   : (r.solver === 'exact' ? 'exact' : 'fallback'),
                 f(r.mean_solve_ms, 4), f(r.max_solve_ms, 4), f(r.budget_frac_pct, 4)]),
    [800, 1200, 700, 1200, 1700, 1200, 2100], { left: false, size: 13 }),
  CAP(`Table XI. Solve time against action-set size and horizon, at the default dead `
  + `time. Rows marked “fallback” exceeded the exact solver’s node budget and were solved `
  + `by the greedy completion, so their times are not times for Algorithm 2. Synthetic traces, `
  + `2 families × 4 seeds per row. The control interval is 5 s.`),
  P(`At the configuration used throughout the paper the mean solve time is ${f(e9def.mean_solve_ms, 3)} ms, which is `
  + `${f(e9def.budget_frac_pct, 4)} % of the control interval. The largest configuration tested, Δmax = ${e9big.delta_max} with a `
  + `${e9big.H}-step horizon, costs ${f(e9big.mean_solve_ms, 3)} ms on average and ${f(e9max, 2)} ms at worst. Fig. 12(b) `
  + 'plots measured time against the two parameters. The cost is set at configuration time rather '
  + 'than by the data: for a given action set, horizon and dead time the search has a fixed shape, '
  + 'and the spread across seeds reflects only how often admissibility prunes a branch.'),
  P('The frontier matters as much as the timings. Exact solution is affordable in two regions — '
  + 'short horizons, where few actions reach inside the horizon at all, and horizons long enough '
  + 'relative to the dead time that the pending queue stops distinguishing states — but not in the '
  + 'band between them, and not at large action sets, where the branching factor dominates both. '
  + 'The default configuration sits inside the affordable region, which is why exact results can be '
  + 'reported throughout this paper; a pool with a much larger rate limit would not allow it.'),
  P(`The consequence for the paper’s argument is narrower than it was before the solver was `
  + `corrected. Relaxation saves real compute: at the default configuration it is the difference `
  + `between ${f(e9def.mean_solve_ms, 1)} ms and a few tenths of a millisecond per tick. What it does not save is `
  + `anything that is scarce. The saving is ${f(e9def.budget_frac_pct, 2)} % of a control interval that the controller `
  + `spends idle, set against a behavioural difference on a large fraction of control decisions. `
  + `Relaxation at container-pool scale is therefore not a performance optimisation but a modelling `
  + `choice, and one whose cost is now measured. The argument does not carry to cluster-scale `
  + `problems with hundreds of coupled decision variables: there the exact search is not available `
  + `at all, as the frontier above already shows in miniature.`),
  img('fig_solvetime.png', 6.9),
  CAP('Fig. 12. Solve-time scaling against action-set size and horizon. Points beyond '
  + 'the exact solver’s node budget were produced by the greedy completion and are not '
  + 'times for Algorithm 2. Synthetic traces.'),
];

// ============================================================ V-H cold start
function cs(m, tau) { return one(E12, r => r.method === m && Math.abs(r.tau_c_s - tau) < 1e-9); }
const taus12 = [...new Set(E12.map(r => r.tau_c_s))].sort((a, b) => a - b);
const tauLo = taus12[0], tauHi = taus12[taus12.length - 1];

const coldstart = [
  H2('J. Robustness to Actuation Delay'),
  P('Section V-A varied the horizon at fixed cold start. The complementary experiment fixes the '
  + 'horizon at a value that satisfies (6) everywhere and varies the cold start instead, which '
  + 'isolates the controller’s tolerance of actuation delay from the question of whether the '
  + 'horizon is adequate. Table XIII reports it and Fig. 14 shows both panels.'),
  mkTable(
    ['τc (s)', 'Exact MPC\nviol. (%)', 'Rounded\nviol. (%)', 'Reactive\nviol. (%)',
     'Exact\nactions', 'Rounded\nactions', 'Reactive\nactions'],
    taus12.map(t => [
      String(Math.round(t)),
      f(cs('pact', t).sla_mean, 2), f(cs('rounded', t).sla_mean, 2), f(cs('reactive', t).sla_mean, 2),
      f(cs('pact', t).act_mean, 1), f(cs('rounded', t).act_mean, 1), f(cs('reactive', t).act_mean, 1),
    ]),
    [900, 1400, 1300, 1300, 1200, 1300, 1300], { left: false, size: 13 }),
  CAP(`Table XIII. Sensitivity to cold-start latency with the horizon fixed at H = 30 (150 s lookahead), `
  + `which satisfies (6) at every value shown. Synthetic traces, 3 families × 5 seeds = ${cs('pact', tauLo).n_runs} runs per cell.`),
  P('With an adequate horizon the integer-exact controller is close to insensitive to actuation delay '
  + `over the practical range. Violations stay between ${f(cs('pact', 5).sla_mean, 2)} % and ${f(cs('pact', 60).sla_mean, 2)} % as the cold start grows from `
  + `${Math.round(tauLo)} s to 60 s, a factor of twelve in delay. The reactive controller degrades steadily over the same `
  + `range, from ${f(cs('reactive', 5).sla_mean, 2)} % to ${f(cs('reactive', 60).sla_mean, 2)} %, and the rounded controller degrades faster still, from `
  + `${f(cs('rounded', 5).sla_mean, 2)} % to ${f(cs('rounded', 60).sla_mean, 2)} %. At the longest delay tested, ${Math.round(tauHi)} s, the three sit at `
  + `${f(cs('pact', tauHi).sla_mean, 2)} %, ${f(cs('reactive', tauHi).sla_mean, 2)} % and ${f(cs('rounded', tauHi).sla_mean, 2)} % respectively.`),
  P('This is the experiment that demonstrates what the ablation of Section V-E could not. In that '
  + 'ablation, disabling the dead-time term changed nothing, because the horizon remained long enough '
  + 'that the optimiser could compensate implicitly. Here the dead time itself is the variable, and '
  + 'the separation between the predictive and reactive controllers grows monotonically with it. '
  + 'Dead-time tolerance is therefore a property of the predictive formulation as a whole rather than '
  + 'of the explicit delay term in (5), and the paper’s claim is stated at that level.'),
  P('The churn panel adds a second observation. The rounded controller’s action count is '
  + `essentially flat at roughly ${f(cs('rounded', 45).act_mean, 0)} per run regardless of delay, because its oscillation is driven `
  + 'by fractional rounding rather than by the plant. The exact controller’s action count varies with '
  + 'delay, as a controller responding to the plant should.'),
  img('fig_coldstart.png', 6.9),
  CAP('Fig. 14. Service level and churn against cold-start latency at a fixed adequate horizon. '
  + 'Synthetic traces.'),
];

// ============================================================ V-I per workload
function pw(wl, m) { return one(E10, r => r.workload === wl && r.method === m); }
const wls = [...new Set(E10.map(r => r.workload))];

const perWorkload = [
  H2('K. Per-Workload Behaviour'),
  P('The aggregate figures of Section V-D average over five demand shapes that stress the controller '
  + 'differently. Table XIV disaggregates them, and the variation is large enough that the aggregate '
  + 'alone would be misleading.'),
  mkTable(
    ['Workload', 'Method', 'Violation (%)', 'Over-prov. (%)', 'Actions', 'Oscillations', 'p95 (ms)'],
    wls.flatMap(wl => [['pact-oracle', 'Exact MPC'], ['rounded-oracle', 'Rounded'],
                       ['reactive', 'Reactive'], ['static', 'Static']]
      .map(([m, lab], i) => {
        const r = pw(wl, m);
        return [i === 0 ? wl.charAt(0).toUpperCase() + wl.slice(1) : '', lab,
                `${f(r.sla_mean, 2)} ± ${f(r.sla_std, 2)}`,
                `${f(r.over_mean, 1)} ± ${f(r.over_std, 1)}`,
                f(r.act_mean, 1), f(r.osc_mean, 1), f(r.p95_mean, 0)];
      })),
    [1300, 1300, 1700, 1700, 900, 1200, 900], { left: false, size: 12 }),
  CAP(`Table XIV. Per-workload breakdown. Synthetic traces, 10 seeds per row, oracle predictor; `
  + 'values are mean ± one standard deviation.'),
  P('The burst family separates the methods most sharply. There the rounded controller reaches '
  + `${f(pw('burst', 'rounded-oracle').sla_mean, 2)} % violations against ${f(pw('burst', 'pact-oracle').sla_mean, 2)} % for the exact controller, the largest gap in the table, `
  + 'because a burst drives the pool through many integer boundaries in quick succession and every '
  + 'crossing is an opportunity for the rounding error to change the decision. The diurnal and ramp '
  + 'families, whose demand changes slowly relative to the control interval, show almost no separation '
  + `at all — ${f(pw('diurnal', 'rounded-oracle').sla_mean, 2)} % against ${f(pw('diurnal', 'pact-oracle').sla_mean, 2)} % on diurnal traffic. The advantage of exact solution is `
  + 'therefore concentrated in workloads with fast transitions, which is also where an auto-scaler is '
  + 'most needed and least likely to perform well.'),
  P('The sawtooth family is the one case where the exact controller is worse, at '
  + `${f(pw('sawtooth', 'pact-oracle').sla_mean, 2)} % against ${f(pw('sawtooth', 'rounded-oracle').sla_mean, 2)} %, and the standard deviation across seeds is correspondingly `
  + `large at ${f(pw('sawtooth', 'pact-oracle').sla_std, 2)}. A sawtooth resets demand discontinuously every ninety ticks; the churn penalty `
  + 'resists the repeated scale-down that the shape demands, and the controller spends part of each '
  + 'period recovering. This is a genuine failure mode of the churn penalty under periodic '
  + 'discontinuity and is reported as such rather than averaged away.'),
  img('fig_per_workload.png', 6.9),
  CAP('Fig. 15. Per-workload service level and capacity cost. Whiskers are one standard deviation '
  + 'over 10 seeds. Synthetic traces.'),
];

// ============================================================ V-J decisions
function dec(m) { return one(E11, r => r.method === m); }
const dPo = dec('pact-oracle'), dPp = dec('pact-persist');
const dRo = dec('rounded-oracle'), dRe = dec('reactive');

const decisions = [
  H2('L. Decision-Level Analysis'),
  P('Treating each tick as a three-way classification into scale-up, hold and scale-down against the '
  + 'requirement implied by the observed load gives a finer view than the aggregate rates. Table XV '
  + 'reports it with bootstrap intervals and Fig. 16 shows it. The numbers are low in absolute terms '
  + 'for every method, and the reason is worth stating because it bears on how such metrics should be '
  + 'read.'),
  mkTable(
    ['Method', 'Scale-up recall', 'Scale-up precision', 'Hold F1', 'Accuracy'],
    [[ 'Exact MPC (oracle)', `${f(dPo.scaleup_recall_mean, 3)} [${f(dPo.scaleup_recall_lo, 3)}, ${f(dPo.scaleup_recall_hi, 3)}]`,
       `${f(dPo.scaleup_precision_mean, 3)} [${f(dPo.scaleup_precision_lo, 3)}, ${f(dPo.scaleup_precision_hi, 3)}]`,
       f(dPo.hold_f1_mean, 3), f(dPo.accuracy_mean, 3)],
     [ 'Exact MPC (persistence)', `${f(dPp.scaleup_recall_mean, 3)} [${f(dPp.scaleup_recall_lo, 3)}, ${f(dPp.scaleup_recall_hi, 3)}]`,
       `${f(dPp.scaleup_precision_mean, 3)} [${f(dPp.scaleup_precision_lo, 3)}, ${f(dPp.scaleup_precision_hi, 3)}]`,
       f(dPp.hold_f1_mean, 3), f(dPp.accuracy_mean, 3)],
     [ 'Relaxed + rounded', `${f(dRo.scaleup_recall_mean, 3)} [${f(dRo.scaleup_recall_lo, 3)}, ${f(dRo.scaleup_recall_hi, 3)}]`,
       `${f(dRo.scaleup_precision_mean, 3)} [${f(dRo.scaleup_precision_lo, 3)}, ${f(dRo.scaleup_precision_hi, 3)}]`,
       f(dRo.hold_f1_mean, 3), f(dRo.accuracy_mean, 3)],
     [ 'Reactive threshold', `${f(dRe.scaleup_recall_mean, 3)} [${f(dRe.scaleup_recall_lo, 3)}, ${f(dRe.scaleup_recall_hi, 3)}]`,
       `${f(dRe.scaleup_precision_mean, 3)} [${f(dRe.scaleup_precision_lo, 3)}, ${f(dRe.scaleup_precision_hi, 3)}]`,
       f(dRe.hold_f1_mean, 3), f(dRe.accuracy_mean, 3)]],
    [2000, 2200, 2200, 1200, 1200], { left: false, size: 13 }),
  CAP(`Table XV. Per-tick decision classification against the observed requirement. Synthetic traces, `
  + `${dPo.n_runs} runs per row; intervals are 95 % bootstrap.`),
  P('Scale-up recall is low for every method, between '
  + `${f(Math.min(dPo.scaleup_recall_mean, dPp.scaleup_recall_mean, dRo.scaleup_recall_mean, dRe.scaleup_recall_mean), 3)} and `
  + `${f(Math.max(dPo.scaleup_recall_mean, dPp.scaleup_recall_mean, dRo.scaleup_recall_mean, dRe.scaleup_recall_mean), 3)}. This is a property of the ground-truth definition rather than of the controllers. `
  + 'The requirement series is computed per tick from the instantaneous arrival rate and therefore '
  + 'changes on most ticks, including by a single replica in response to noise, whereas a well-behaved '
  + 'controller deliberately does not track every such change. A controller that matched this ground '
  + 'truth tick for tick would be the oscillating controller the design is intended to avoid. The '
  + 'metric is reported for completeness and because it is standard in the literature, but it should '
  + 'not be read as a quality ranking, and the paper does not use it as one.'),
  P('The one comparison it does support is between predictors at fixed control law. Moving from '
  + `persistence to perfect foresight changes scale-up recall from ${f(dPp.scaleup_recall_mean, 3)} to ${f(dPo.scaleup_recall_mean, 3)} and precision from `
  + `${f(dPp.scaleup_precision_mean, 3)} to ${f(dPo.scaleup_precision_mean, 3)}. The persistence predictor scores higher on both, which at first appears `
  + 'paradoxical. It follows from the same definitional issue: persistence reacts to the most recent '
  + 'observation and therefore produces decisions that correlate with a per-tick ground truth derived '
  + 'from that same observation, while the oracle acts on demand that has not yet arrived and is '
  + 'scored as wrong until it does. The aggregate service-level figures in Table IV, where the oracle '
  + 'is ahead, are the more meaningful comparison.'),
  img('fig_decisions.png', 6.9),
  CAP('Fig. 16. Per-tick decision classification with 95 % bootstrap intervals. Synthetic traces.'),
];

// ============================================================ deployment guidance
const guidance = [
  H2('N. Configuration Guidance'),
  P('The measurements support a short configuration procedure, stated here because the parameters '
  + 'interact and the order matters.'),
  BUL('Measure the cold-start latency of the deployed image, including image pull on a cold cache, '
    + 'readiness probing and load-balancer registration. This single measurement determines the '
    + 'horizon through (6) and is the one parameter that cannot be guessed.'),
  BUL(`Set the horizon to the smallest value satisfying HΔt ≥ τc, then add one interval of slack. `
    + 'Section V-A shows the penalty for an under-sized horizon is severe and the benefit of '
    + 'oversizing it is negligible, so the asymmetry favours erring long.'),
  BUL(`Set the cooldown next. Section V-G shows it is the dominant stability control; ${cd60.value} s `
    + 'was adequate on every workload tested and longer values continued to help.'),
  BUL('Set the churn weight to approximately unity. Below that value stability degrades; above it '
    + 'nothing changes.'),
  BUL('Leave the deadband at zero or one and the cost asymmetry at any moderate value. Neither was '
    + 'found to matter, and a deadband of two or more trades service level for stability that the '
    + 'cooldown already provides more cheaply.'),
  P('The procedure assumes a single replica pool with the plant characteristics described in Section '
  + 'IV-A and should be re-validated before transfer to a different deployment, particularly one in '
  + 'which several services share hosts.'),
];


// ============================================================ V-I exact vs greedy
const G15 = E15.paper_config;
const gcAt = v => E8C.find(r => Math.abs(r.value - v) < 1e-9);
const gc0 = gcAt(0.0);
const gc1 = gcAt(1.0);
const gcw = E8C.reduce((a, b) =>
  Math.abs(b.greedy_sla - b.exact_sla) > Math.abs(a.greedy_sla - a.exact_sla) ? b : a);
// disagreement by exact action, computed from the per-tick record
const byU = {};
E15T.forEach(r => {
  const u = Number(r.exact_u);
  byU[u] = byU[u] || { n: 0, d: 0 };
  byU[u].n += 1;
  if (Number(r.agree) === 0) byU[u].d += 1;
});
const peakU = Object.keys(byU).map(Number).filter(u => byU[u].n >= 20)
  .sort((a, b) => (byU[b].d / byU[b].n) - (byU[a].d / byU[a].n))[0];
const peakPct = 100 * byU[peakU].d / byU[peakU].n;
const bigUp = Object.keys(byU).map(Number).filter(u => u >= 3);
const bigUpN = bigUp.reduce((acc, u) => acc + byU[u].n, 0);
const bigUpD = bigUp.reduce((acc, u) => acc + byU[u].d, 0);
const bigUpPct = 100 * bigUpD / Math.max(bigUpN, 1);
const disagree = (100.0 - G15.agree_pct);
const exactGap = [
  H2('I. Exact Versus Greedy-Completion Solving'),
  P('Section V-H establishes that exact solving is affordable. This section asks the prior '
  + 'question of whether it is necessary, by comparing Algorithm 2 against the cheaper solver it '
  + 'replaced: one that enumerates the admissible first actions but completes each horizon by '
  + 'tracking demand myopically. Both solvers were given the same '
  + `${G15.ticks} plant states, drawn to span the reachable range of replica count, in-flight queue, `
  + 'previous action and forecast slope, and both were scored with the same objective (7)-(9).'),
  mkTable(
    ['Quantity', 'Value'],
    [
      ['Dead time τ, horizon H, tracking-relevant actions m',
       `${G15.tau}, ${G15.horizon}, ${G15.m}`],
      ['Plant states compared', `${G15.ticks}`],
      ['First actions in agreement', `${G15.agree} (${f(G15.agree_pct, 2)} %)`],
      ['First actions in disagreement', `${G15.ticks - G15.agree} (${f(disagree, 2)} %)`],
      ['Greedy completion attains a worse objective',
       `${G15.greedy_worse} (${f(G15.greedy_worse_pct, 2)} %)`],
      ['Greedy completion attains a better objective', `${G15.greedy_better}`],
      ['Mean objective excess of greedy completion', `${f(G15.cost_excess_mean, 3)}`],
      ['Largest objective excess of greedy completion', `${f(G15.cost_excess_max, 2)}`],
      ['Mean relative objective excess', `${f(G15.rel_excess_mean_pct, 3)} %`],
      ['Exact solve time, median / p95 / max',
       `${f(G15.exact_p50_ms, 2)} / ${f(G15.exact_p95_ms, 2)} / ${f(G15.exact_max_ms, 2)} ms`],
      ['Greedy solve time, median / p95 / max',
       `${f(G15.greedy_p50_ms, 3)} / ${f(G15.greedy_p95_ms, 3)} / ${f(G15.greedy_max_ms, 3)} ms`],
    ],
    [6200, 3100], { left: true }),
  CAP('Table XII. Algorithm 2 against enumeration with greedy horizon completion, on '
  + 'the configuration used throughout this paper. Objective excess is in units of the cost (7).'),
  P(`The two solvers choose the same action on ${f(G15.agree_pct, 2)} % of states and differ on the remaining `
  + `${f(disagree, 2)} %. On ${f(G15.greedy_worse_pct, 2)} % of states the greedy completion lands on a plan of strictly higher `
  + 'cost, and on none does it beat Algorithm 2 — the expected signature of an exact reference, and '
  + 'the reason the comparison is reported in this direction rather than as a symmetric '
  + 'disagreement rate.'),
  P('Two features of the gap matter more than its size. It is one-sided, so the greedy completion '
  + 'is a relaxation of the same problem rather than a different trade-off; and it is concentrated '
  + `rather than uniform, with a mean relative excess of ${f(G15.rel_excess_mean_pct, 3)} % against a maximum of `
  + `${f(G15.rel_excess_max_pct, 1)} %. The states where it errs are those whose optimal plan ends the horizon on a large `
  + 'action, which is precisely where zeroing the tail misprices the churn term. A controller built '
  + 'on the greedy completion is therefore not uniformly slightly worse; it is almost always '
  + 'identical and occasionally substantially wrong.'),
  img('fig_exact_vs_greedy.png', 6.9),
  CAP('Fig. 13. Algorithm 2 against greedy horizon completion over the same '
  + 'plant states. (a) Disagreement is not uniform over the action space: it peaks on moderate '
  + 'scale-downs and nearly vanishes on large scale-ups. (b) The objective excess is one-sided by '
  + 'construction; no tick favours the greedy completion. (c) Solve time for both, against the '
  + 'control interval.'),
  P('The disagreement is structured rather than scattered, as Fig. 13(a) shows. It peaks at '
  + `u*(t) = ${peakU}, where the two solvers choose differently on ${f(peakPct, 1)} % of ticks, and falls `
  + `to ${f(bigUpPct, 1)} % once the exact action is a scale-up of three or more. `
  + 'That shape follows from the mechanism: a large scale-up saturates '
  + 'the rate limit, so every completion agrees on it, whereas a moderate scale-down leaves room '
  + 'for the horizon to be finished several ways and the mispriced tail decides between them. The '
  + 'cases the approximation gets wrong are therefore the ordinary ones, not the extreme ones. '
  + 'Fig. 13(b) shows the excess is one-sided, and Fig. 13(c) the two solve-time distributions '
  + 'against the control interval.'),
  P('One consequence reaches past the solver and into how the controller is tuned. The '
  + 'control-parameter sweep of Section V-G was run once under each solver, with the same '
  + 'workloads, the same seeds and the same configuration, so the pair isolates what approximating '
  + 'the solve does to a tuning conclusion. Under the greedy completion, raising the churn weight '
  + `to unity drove the violation rate to ${f(gc1.greedy_sla, 3)} % and held it there across every larger value `
  + `tested. Under Algorithm 2 the same sweep sits near ${f(gc1.exact_sla, 2)} %, and the largest divergence over `
  + `the sweep is ${f(gcw.greedy_sla, 3)} % against ${f(gcw.exact_sla, 2)} % at s = ${gcw.value}.`),
  P('The qualitative reading of that sweep is unchanged: raising s still improves service level, '
  + `capacity and oscillation together, from ${f(gc0.exact_over, 1)} % over-provisioning and ${f(gc0.exact_osc, 1)} oscillations at `
  + `s = 0 to ${f(gc1.exact_over, 1)} % and ${f(gc1.exact_osc, 1)} at s = 1. What was not a property of the controller is the size `
  + 'of the effect. The mechanism is the one Section III-E sets out: greedy completion zeroes the '
  + 'tail, a zero tail misprices precisely the term s scales, and raising s magnifies the '
  + 'mispricing. An approximate solver therefore misleads most where the parameter being tuned is '
  + 'the one the approximation distorts, which is not where a reader would think to look for it.'),
  P(`Exactness costs roughly ${(G15.exact_p50_ms / G15.greedy_p50_ms).toFixed(0)} times the compute of the greedy completion. `
  + `That multiplier is the real price of the claim. In absolute terms the worst observed solve took ${f(G15.exact_max_ms, 1)} ms, or `
  + `${f(100.0 * G15.exact_max_ms / 5000.0, 2)} % of the five-second control interval, so the multiplier is large and the `
  + 'absolute cost is not.'),
  P('That cost is not always payable. The reduction in Section III-E is strong when the dead time '
  + 'is long relative to the horizon, and the memoised recursion is cheap when the dead time is '
  + 'short enough for the pending queue to collapse distinct prefixes onto one state. Between the '
  + 'two lies a band — a queue long enough to keep states distinct and a horizon long enough to '
  + 'make m large — where neither applies, and there the controller falls back to the greedy '
  + 'completion rather than stall the loop. Every result at the configuration used throughout this '
  + 'paper is solved exactly. The horizon sweep of Section V-A is the one experiment that crosses '
  + (E5X
     ? `the band: ${E5X.exact} of its ${E5X.cells} cells are solved exactly and ${E5X.fallback} are not, with each run `
       + 'recording which solver produced it. '
       + (E5X.edgeExact
          ? 'The fallbacks all sit well above the boundary, where the controller is already at its '
            + 'floor; every cell immediately either side of HΔt = τc is solved exactly, so the '
            + 'transition itself is measured without the approximation.'
          : 'Some of the fallbacks sit next to the boundary, which is noted where those cells are read.')
     : 'the band, and each run records which solver produced it.')),
];

module.exports = { litSection, algos, sensitivity, solvetime, exactGap,
                   coldstart, perWorkload, decisions, guidance };
