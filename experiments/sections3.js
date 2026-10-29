/* Deeper methodology: per-module implementation detail and the plant model. */
const B = require('./build_paper.js');
const { P, RUNS, H2, H3, CAP, BUL, EQ, sub, sup, mkTable, MAN, f } = B;
const d = require('docx');
const { MathRun, MathFraction } = d;

// ---------------------------------------------------------- module detail
const moduleDetail = [
  H2('F. Module-Level Implementation'),
  P('The controller is implemented as five modules with narrow interfaces, so that each can be '
  + 'disabled independently for the ablation of Section V-E and so that the measurement harness can '
  + 'substitute a predictor without touching the control path. This subsection records what each '
  + 'module does in the implementation, at the level of detail needed to reproduce it.'),

  H3('Telemetry and feature construction'),
  P('The collector exposes a single method returning the observation vector of (1). Container '
  + 'runtimes report cumulative counters rather than rates: processor time arrives as consumed '
  + 'nanoseconds and network and block activity as byte totals. Each is differenced against the '
  + 'previous sample and divided by the interval before it carries meaning as load. Counter resets, '
  + 'which occur when a container restarts, would otherwise produce a large negative rate; the '
  + 'implementation detects a decrease and treats the new value as the first sample of a fresh '
  + 'counter rather than differencing across the discontinuity.'),
  P('Each channel is then smoothed by a first-order exponentially weighted filter and mapped to the '
  + 'unit interval using per-channel bounds estimated once and frozen. Freezing matters: if the '
  + 'normalisation adapted online, the same raw utilisation would map to different model inputs at '
  + 'different times and the predictor would be chasing a moving target. The implementation enforces '
  + 'this by refusing to refit the normaliser after it has been frozen, which converts a subtle '
  + 'statistical error into an immediate exception.'),
  P('Two derived channels are appended. The first is the per-replica arrival intensity, the arrival '
  + 'rate divided by the current replica count. Without it the predictor would be forecasting a '
  + 'signal that the controller itself modifies, because utilisation falls whenever the controller '
  + 'adds capacity; dividing it out leaves the exogenous demand. The second is the time of day '
  + 'encoded as a sine and cosine pair, which supplies daily periodicity without requiring the model '
  + 'to span a full day of history.'),

  H3('Capacity mapping'),
  P('The mapper is a pure function with no internal state, which makes it trivially testable and '
  + 'means the ablation can replace it without disturbing anything else. For each horizon step it '
  + 'computes the processor-bound requirement from (2), the memory-bound requirement, takes the '
  + 'larger of the two as in (3), and then increases the candidate while the latency estimate of (4) '
  + 'exceeds the objective. That loop terminates because the candidate increases strictly and is '
  + 'bounded above by the pool maximum, a property worth stating because an unbounded search inside '
  + 'a control loop is a liveness hazard rather than merely a performance one.'),
  P('The ceiling in (2) is deliberate. Rounding to nearest would accept a shortfall of up to half a '
  + 'replica as adequate, and at the small pool sizes this work targets half a replica can be several '
  + 'per cent of total capacity. The margin γ is applied before the ceiling rather than after, so '
  + 'that a small margin can still change the integer outcome rather than being rounded away.'),

  H3('Gate ordering'),
  P('The gates apply in a fixed order: round the relaxed value, clip to the rate limit, clip to the '
  + 'pool bounds, apply the deadband, then apply the cooldown. The order separates feasibility from '
  + 'policy. The first three steps guarantee the action is admissible with respect to the hard '
  + 'constraints of (9); the last two decide whether an admissible action is worth taking. Reversing '
  + 'them would allow a deadband or cooldown decision to be taken on an action that the bounds were '
  + 'about to modify, so that the gate would be reasoning about a value never applied.'),
  P('The implementation records the intermediate value at each stage rather than only the final '
  + 'action. This is not instrumentation for its own sake: when a controller produces an unexpected '
  + 'replica count, the question is always which stage changed the decision, and a single logged '
  + 'number cannot answer it.'),

  H3('Drift monitoring'),
  P('The monitor compares the replica count the observed load actually required at tick t against the '
  + 'count that was predicted for t by the forecast issued at t − τ. Implementing that comparison '
  + 'requires retaining each issued horizon keyed by both its target tick and its issue tick, because '
  + 'the relevant prediction is a specific element of a specific past forecast rather than the most '
  + 'recent one. An off-by-one here is silent: the margin still adapts, it simply adapts against the '
  + 'wrong comparison, and nothing in the output distinguishes the two. The implementation therefore '
  + 'keys the buffer explicitly and the test suite verifies the alignment against hand-computed '
  + 'values.'),
  P('The update itself is the clipped integral of (11). Clipping is what prevents a single anomalous '
  + 'period from driving the margin to a value that is expensive to unwind, and it is also what makes '
  + 'the mechanism deterministic in the sense this paper uses: the margin is a scalar with known '
  + 'bounds and a known update rule, inspectable in a log line, rather than a posterior that must be '
  + 'sampled.'),

  H3('Actuation'),
  P('The simulated actuator applies the action to the plant with the same bounds and rate limit the '
  + 'controller assumed, which closes the loop consistently. The repository also contains a Docker '
  + 'backend that issues scale commands and reconciles the observed container count against the '
  + 'desired one, polling until they agree or a timeout expires. That backend is idempotent, so '
  + 'issuing the current count is a no-op, and it is asynchronous, so reconciliation never blocks a '
  + 'control tick. It was not exercised in this evaluation because no container runtime was '
  + 'available; Fig. 1 marks it accordingly.'),
];

// ---------------------------------------------------------- plant model
const plantDetail = [
  H2('C. Plant Model'),
  P('The simulated plant is the component that determines how far these results transfer, so its '
  + 'assumptions are stated explicitly rather than left to the source.'),
  P('The pool maintains two sets, ready replicas and pending replicas. An action that adds capacity '
  + 'places the new replicas in the pending set with a release tick τ ahead of the present; they '
  + 'contribute no service capacity until that tick arrives. An action that removes capacity takes '
  + 'effect immediately, which is the asymmetry a real container runtime exhibits, since stopping a '
  + 'process is fast and starting one is not. Offered load is divided among ready replicas only.'),
  P('Service is modelled as a multi-server queue whose per-replica rate is constant. Tail latency '
  + 'follows the heavy-traffic approximation of (4) with a p95 obtained by assuming an exponential '
  + 'sojourn distribution, which is exact for a single-server Markovian queue and an approximation '
  + 'otherwise. Memory is modelled as a fixed per-replica baseline plus a term proportional to '
  + 'utilisation. The approximation diverges as utilisation approaches unity; Section IV-A describes '
  + 'the censoring applied there and every reported mean carries its censoring rate.'),
  P('Three properties of real deployments are absent and each bounds the conclusions. Replica '
  + 'start-up is a fixed latency rather than a distribution, so the variance that image-pull '
  + 'contention and scheduler queuing introduce is not represented. Replicas are identical, so '
  + 'heterogeneous hosts and noisy neighbours are absent. Requests are not queued across the '
  + 'capacity change, so the transient when a new replica joins and the load balancer redistributes '
  + 'connections is not modelled. All three would make the plant harder to control, and the measured '
  + 'service-level figures should therefore be read as optimistic in absolute terms even though the '
  + 'comparisons between controllers, which share the plant, remain valid.'),
];

module.exports = { moduleDetail, plantDetail };

// ---------------------------------------------------------- inference trace
const E14 = B.csv('e14_inference_trace.csv');
const disagreeN = E14.filter(r => r.mpc_u !== r.rounded_u).length;
const blockedN = E14.filter(r => r.cooldown_blocked === 'True' || r.cooldown_blocked === true).length;
const meanErr = E14.reduce((s, r) => s + r.error_vs_required, 0) / E14.length;

const inferenceTrace = [
  H2('N. Decision Trace'),
  P('Aggregate metrics describe behaviour but not mechanism. Table XVI reproduces sixteen '
  + 'consecutive control ticks from a single run, taken from the rise of the first burst on the '
  + 'burst trace, with every intermediate quantity the controller computes. It is included because '
  + 'the differences the aggregate tables report are visible here as individual decisions, and '
  + 'because an auto-scaler that cannot be read tick by tick cannot be debugged.'),
  mkTable(
    ['Tick', 'λ\n(req/s)', 'Observed\nu', 'Forecast\nu at h=12', 'Demand\nÑ(t+1)',
     'MPC\nu*', 'Rounded\nu', 'After\ngates', 'Applied\nu', 'N(t)',
     'Required\n(ground truth)', 'Excess'],
    E14.map(r => [
      String(r.tick), f(r.arrival_rate, 0), f(r.observed_util, 3), f(r.forecast_u_h12, 3),
      String(r.demand_h1), String(r.mpc_u), String(r.rounded_u), String(r.after_gates),
      String(r.applied_u), String(r.n_desired), String(r.ground_truth_required),
      (r.error_vs_required > 0 ? '+' : '') + String(r.error_vs_required),
    ]),
    [550, 700, 750, 900, 750, 600, 700, 650, 650, 550, 950, 650], { left: false, size: 11 }),
  CAP('Table XVI. Sixteen consecutive control ticks from the burst trace, seed 1001, oracle '
  + 'predictor. The ground-truth column is the replica count the instantaneous arrival rate required '
  + 'at the target utilisation; it is computed after the fact and is not available to the controller. '
  + 'The excess column is N(t) minus that requirement.'),
  P(`The two controllers disagree on ${disagreeN} of the ${E14.length} ticks shown, which is consistent with the rates in `
  + 'Table V. The pattern of disagreement is informative: the rounded controller proposes a '
  + 'scale-down on most ticks where the exact controller proposes holding. Its relaxed solution sits '
  + 'slightly below the current count because the instantaneous demand is slightly below it, and '
  + 'rounding converts that fractional preference into a whole replica. The exact controller, '
  + 'evaluating the same horizon under the integrality constraint, finds that removing a replica '
  + 'costs more over the horizon than it saves on the current tick.'),
  P(`The gate column shows the cooldown active on ${blockedN} ticks, each time converting a proposed scale-down to a `
  + 'hold. Ticks 113 through 117 are the clearest case: the optimiser proposes removing a replica on '
  + 'five consecutive ticks and the cooldown suppresses all five, because a scale-up occurred '
  + 'recently. The arrival rate then rises through tick 121, which retrospectively makes every one '
  + 'of those suppressed scale-downs the wrong action. This is the mechanism behind the sensitivity '
  + 'result of Section V-F, where the cooldown dominated stability, seen at the level of the '
  + 'individual decisions it changes.'),
  P(`The excess column averages ${f(meanErr, 2)} replicas over the window, which is the over-provisioning the `
  + 'aggregate tables report expressed per tick. It is positive on almost every tick rather than '
  + 'alternating in sign, which identifies it as a systematic bias of the controller rather than '
  + 'tracking noise. The asymmetric tracking weight of (8) and the cooldown both push in that '
  + 'direction, and Section V-F shows the cooldown is the larger contributor.'),
  P('One caveat applies to reading this table. The ground-truth column is derived from the '
  + 'instantaneous arrival rate, so it changes on nearly every tick, including by a single replica '
  + 'in response to noise. A controller that matched it exactly would be the oscillating controller '
  + 'the design is intended to avoid, and the excess column should be read as a measure of '
  + 'conservatism rather than of error. This is the same definitional caveat that applies to the '
  + 'decision classification of Section V-J.'),
];

module.exports.inferenceTrace = inferenceTrace;

// ---------------------------------------------------------- validity framing
const validity = [
  H2('A. Construct Validity'),
  P('Construct validity asks whether the quantities measured correspond to the properties claimed. '
  + 'Two constructs in this paper are imperfect and the imperfection bounds the claims that rest on '
  + 'them.'),
  P('The service-level violation rate counts ticks whose modelled tail latency exceeds the '
  + 'objective. Real service-level objectives are evaluated over windows and against measured '
  + 'request latencies, not over control ticks against a queueing approximation. A controller that '
  + 'performs well on this construct would perform well on the real one only to the extent the '
  + 'approximation holds, which Section IV-C argues it does in the moderate-utilisation regime and '
  + 'does not near saturation, where the measure is censored.'),
  P('The ground-truth replica requirement, used by both the decision classification of Section V-J '
  + 'and the excess column of Table XVI, is computed from the instantaneous arrival rate. It '
  + 'therefore treats noise as demand and changes on nearly every tick. It is an adequate construct '
  + 'for measuring systematic bias, which is how this paper uses it, and an inadequate one for '
  + 'ranking controllers by decision quality, which is why the classification results are reported '
  + 'but not used to support any claim.'),
  H2('B. Internal Validity'),
  P('Internal validity asks whether the measured differences are caused by the manipulated variable. '
  + 'The experimental design supports this reasonably well. Every method sees the identical trace '
  + 'for a given family and seed, so comparisons are paired and the Wilcoxon test is appropriate. '
  + 'Each ablation and sweep changes exactly one component or parameter with the others held fixed. '
  + 'The plant is shared across methods within a run, so differences cannot arise from plant '
  + 'variation.'),
  P('One threat remains. The methods operate at different default utilisations, so the aggregate '
  + 'comparison of Table VII compares points rather than frontiers. This is the reason the '
  + 'operating-point sweep of Section V-C exists, and conclusions about service level are drawn from '
  + 'that sweep rather than from the aggregate table.'),
  H2('C. External Validity'),
  P('External validity asks how far the results transfer, and it is where this evaluation is '
  + 'weakest. The plant is simulated, the workloads are synthetic, the learned forecaster was never '
  + 'run, and a single replica pool was controlled. Each is enumerated in the following subsection '
  + 'with the specific claim it bounds.'),
  P('One form of transfer is better supported than the others. The horizon condition of (6) is a '
  + 'statement about the relationship between two latencies, and the experiment confirms the '
  + 'transition occurs where that relationship predicts rather than at a fixed horizon. A result '
  + 'that tracks a parameter in this way is more likely to transfer than one that holds only at the '
  + 'values tested, because the mechanism rather than the magnitude is what has been identified. '
  + 'The integrality result is in the opposite position: the magnitude of the rounding gap depends '
  + 'on pool size, demand volatility and the operating point, and Table V shows it varying by more '
  + 'than a factor of three across pool sizes alone.'),
];

module.exports.validity = validity;
