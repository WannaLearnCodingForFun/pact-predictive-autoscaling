/* Section V-Q: the recorded WorldCup98 trace and the trained forecaster.
 * Every value and every survive / fail verdict below is computed from the
 * result files at build time; the rules that produce a verdict are stated in
 * the text next to the table that reports it.
 */
const B = require('./build_paper.js');
const { P, RUNS, H2, H3, CAP, img, mkTable, E3, E16, E17, E17B, E18, E19, E20,
        WC, WCT, TCN, PROV, f } = B;

const TRUTHY = v => v === true || v === 'True' || v === 'true';
const mean = (a, k) => (a.length ? a.reduce((s, r) => s + Number(r[k]), 0) / a.length : NaN);
const REAL = ['worldcup98', 'worldcup98_testpeak'];
const FAM = { worldcup98: 'WC98/trace', worldcup98_testpeak: 'WC98/test' };
const E4R = B.csv('e4_ablation_runs.csv');
const E1R = B.csv('e1_runs.csv');
const pv = p => (p === '' || p === undefined ? '—' : (Number(p) < 0.001 ? '< 0.001' : f(p, 3)));
const pvH = (r, ref) => (r[`p_holm_vs_${ref}`] === '' && Number(r[`p_vs_${ref}`]) === 1
  ? '1 (identical)' : pv(r[`p_holm_vs_${ref}`]));

// ------------------------------------------------------------ the trace
const paris = (bin) => {
  const s = new Date((WC.t0_utc_epoch_s + 7200 + bin * WC.dt_s) * 1000).toISOString();
  return `${s.slice(8, 10)} June ${s.slice(11, 16)}`;
};
const days = PROV.worldcup98.subset.days;
const dates = PROV.worldcup98.subset.dates_paris_local;
const testFrac = 1 - WC.train_fraction - WC.val_fraction;

// ------------------------------------------------------------ E20 start state
const EARLY = Object.keys(E20[0]).find(k => k.startsWith('violation_ticks_first_'));
const EARLY_N = Number(EARLY.split('_').pop());
const e20 = (w, rule, m) => E20.find(r => r.workload === w && r.start_rule === rule && r.method === m);
const synChanged = (() => {
  const rows = E20.filter(r => !REAL.includes(r.workload) && r.start_rule === 'load_sized'
                              && r.method === 'reactive');
  return { changed: rows.reduce((s, r) => s + r.n_windows_start_changed, 0),
           total: rows.reduce((s, r) => s + r.n_runs, 0) };
})();

// ------------------------------------------------------------ E16 accuracy
const MODELS = ['persistence', 'arima', 'tcn', 'lstm', 'gru', 'bilstm', 'transformer'];
const MNAME = { persistence: 'Persistence', arima: 'ARIMA', tcn: 'TCN (proposed)', lstm: 'LSTM',
                gru: 'GRU', bilstm: 'Bi-LSTM', transformer: 'Transformer' };
const acc = (m, h) => E16.find(r => r.model === m && r.horizon === h);
const HMAX = Math.max(...E16.map(r => r.horizon));
const atH = MODELS.map(m => ({ m, r: acc(m, HMAX) })).filter(x => x.r);
const rankH = [...atH].sort((a, b) => a.r.nmae_pct - b.r.nmae_pct);
const tcnRank = rankH.findIndex(x => x.m === 'tcn') + 1;
const best = rankH[0];
const neural = atH.filter(x => !['persistence', 'arima'].includes(x.m));
const bestNeural = [...neural].sort((a, b) => a.r.nmae_pct - b.r.nmae_pct)[0];
const tcnH = acc('tcn', HMAX);
const perH = acc('persistence', HMAX);
const ariH = acc('arima', HMAX);
const tcn1 = acc('tcn', 1);
const per1 = acc('persistence', 1);
const tcnBeatsPersist = MODELS.length && [...new Set(E16.map(r => r.horizon))]
  .filter(h => acc('tcn', h).nmae_pct < acc('persistence', h).nmae_pct).length;
const nH = new Set(E16.map(r => r.horizon)).size;
const HS = [...new Set(E16.map(r => r.horizon))];
const ariPairs = neural.length * HS.length;
const ariWins = neural.reduce((s, x) => s + HS.filter(h => acc('arima', h).nmae_pct < acc(x.m, h).nmae_pct).length, 0);
const paramSpread = Math.max(...neural.map(x => Math.abs(x.r.n_params - tcnH.n_params) / tcnH.n_params * 100));

// ------------------------------------------------------------ E17 brackets
const MET = { sla_violation_pct: 'Violation (%)', over_provision_pct: 'Over-prov. (%)',
              scale_actions: 'Actions', oscillations: 'Oscillations',
              cost_per_1k_req: 'Cost /1k req', mean_p95_ms: 'Mean p95 (ms)' };
const POS = { between: 'between', below_both: 'better than both', above_both: 'worse than both' };
const nBetween = E17B.filter(r => r.position === 'between').length;
const nBelow = E17B.filter(r => r.position === 'below_both').length;
const nAbove = E17B.filter(r => r.position === 'above_both').length;
const outside = E17B.filter(r => r.position !== 'between');
const degenerate = E17B.filter(r => r.oracle_mean === r.persist_mean && r.trained_mean === r.oracle_mean);
const sum17 = (w, m) => E17.find(r => r.workload === w && r.method === m);
const nWin = w => sum17(w, 'pact-trained').n_runs;
// Trained vs. persistence: significant after Holm, and in which direction.
const isSigP = p => p !== '' && p !== undefined && Number(p) < 0.05;
const sigP = E17B.filter(r => isSigP(r.p_holm_vs_persist));
const sigO = E17B.filter(r => isSigP(r.p_holm_vs_oracle));
const sigPbetter = sigP.filter(r => r.trained_mean < r.persist_mean);
const sigPworse = sigP.filter(r => r.trained_mean > r.persist_mean);
const describe = r => `${FAM[r.workload]}, ${r.controller === 'pact' ? 'exact' : 'rounded'}, `
  + `${MET[r.metric].replace(/ \(.*\)/, '').toLowerCase()}`;

// ------------------------------------------------------------ E21 bias probe, E22 ablation
const E21 = B.csv('e21_bias_probe.csv');
const e21 = (w, c, m) => E21.find(r => r.workload === w && r.controller === c && r.metric === m);
const E21_SHOW = ['sla_violation_pct', 'over_provision_pct', 'scale_actions'];
const E22_METRIC = ['sla_violation_pct', 'over_provision_pct', 'scale_actions'];
const CAP_MET = ['sla_violation_pct', 'over_provision_pct', 'cost_per_1k_req', 'mean_p95_ms'];
const capRows = E21.filter(r => CAP_MET.includes(r.metric) && r.trained_dir !== 0);
const churnRows = E21.filter(r => !CAP_MET.includes(r.metric) && r.trained_dir !== 0);
const capAgree = capRows.filter(r => r.trained_dir === r.biased_dir).length;
const churnAgree = churnRows.filter(r => r.trained_dir === r.biased_dir).length;
const capCloser = capRows.filter(r => Math.abs(r.biased_oracle_mean - r.trained_mean)
  < Math.abs(r.oracle_mean - r.trained_mean)).length;
const E22 = require('fs').existsSync(require('path').join(B.RES, 'e22_margin_ceiling_ablation.csv'))
  ? B.csv('e22_margin_ceiling_ablation.csv') : null;

const PRED = { persist: 'persistence predictor', trained: 'trained TCN' };
const VAR = { no_margin: 'margin', no_safety: 'ceiling' };
const e22 = (w, fc, v, m) => E22.find(r => r.workload === w && r.forecaster === fc && r.variant === v && r.metric === m);
const e22cell = (w, fc, v, m, d) => {
  const r = e22(w, fc, v, m);
  return r.n_windows_changed ? `${Number(r.delta) > 0 ? '+' : ''}${f(r.delta, d)} (${pv(r.p_holm)})` : '0 (identical)';
};
function e22Section() {
  const fcs = ['persist', 'trained'];
  const sigRows = E22.filter(r => r.variant !== 'full' && isSigP(r.p_holm));
  const reproduces = REAL.every(w => fcs.every(fc => E22_METRIC.every(m => {
    const a = arm17(w, `pact-${fc}`);
    return Math.abs(e22(w, fc, 'full', m).full_mean - a.reduce((s, r) => s + Number(r[m]), 0) / a.length) < 1e-3;
  })));
  const line = (w, fc) => {
    const full = e22(w, fc, 'full', 'sla_violation_pct');
    return `On ${FAM[w]} under the ${PRED[fc]} the margin moved in ${full.gamma_moved_runs} of ${full.n_runs} runs and the ceiling `
      + `issued ${full.overrides_total} overrides; disabling the margin changed the outcome in ${Math.max(...E22_METRIC.map(m => e22(w, fc, 'no_margin', m).n_windows_changed))} `
      + `windows and disabling the ceiling in ${Math.max(...E22_METRIC.map(m => e22(w, fc, 'no_safety', m).n_windows_changed))}.`;
  };
  return [
    P('Section V-E ablated the margin and the ceiling only under perfect foresight, where the margin barely moves. '
    + 'The exact controller was therefore re-run on every recorded test window with each mechanism disabled in turn, under '
    + `the persistence and the trained predictors, with paired Wilcoxon tests Holm-corrected across all ${E22.filter(r => r.variant !== 'full').length} `
    + `comparisons (Table XXIII). The full-controller runs ${reproduces ? 'reproduce' : 'do not exactly reproduce'} the corresponding means in Table XX. `
    + REAL.flatMap(w => fcs.map(fc => line(w, fc))).join(' ')
    + ` ${sigRows.length ? `Of these differences, ${sigRows.length} are significant after correction: `
      + sigRows.map(r => `${VAR[r.variant]} off, ${FAM[r.workload]}, ${PRED[r.forecaster]}, ${MET[r.metric].replace(/ \(.*\)/, '').toLowerCase()} `
        + `${Number(r.delta) > 0 ? '+' : ''}${f(r.delta, 2)}`).join('; ') + '.'
      : 'None of the differences is significant after correction.'}`
    + ` The full controller violates on at most ${f(Math.max(...E22.filter(r => r.variant === 'full' && r.metric === 'sla_violation_pct').map(r => r.full_mean)), 3)}\u202f% of ticks here, `
    + 'so these tests have almost no room to show a service-level benefit; what they do show is the capacity each '
    + 'mechanism holds.'),
    mkTable(
      ['Family', 'Predictor', 'Disabled', 'Δ violation (pp)', 'Δ over-prov. (pp)', 'Δ actions'],
      REAL.flatMap(w => fcs.flatMap(fc => ['no_margin', 'no_safety'].map(v => [
        FAM[w], PRED[fc], VAR[v],
        e22cell(w, fc, v, 'sla_violation_pct', 3), e22cell(w, fc, v, 'over_provision_pct', 2),
        e22cell(w, fc, v, 'scale_actions', 1)]))),
      [1400, 1400, 1100, 1700, 1800, 1500], { left: false, size: 12 }),
    CAP(`Table XXIII. Exact MPC with the adaptive margin or the safety ceiling disabled, against the full controller, under `
    + `the two imperfect predictors on recorded traffic, ${E22[0].n_runs} test windows per family. Each cell is the mean paired `
    + 'difference with its Holm-corrected p-value; "identical" means no window changed.'),
  ];
}

// ------------------------------------------------------------ E18 horizon law
function cliffStats(rows) {
  const taus = [...new Set(rows.map(r => r.tau_c_s))].sort((a, b) => a - b);
  const per = taus.map(tau => {
    const rs = rows.filter(r => r.tau_c_s === tau).sort((a, b) => a.H - b.H);
    const below = rs.filter(r => !TRUTHY(r.covers_coldstart));
    const above = rs.filter(r => TRUTHY(r.covers_coldstart));
    return {
      tau, nb: below.length, na: above.length,
      below: mean(below, 'pact_sla'), above: mean(above, 'pact_sla'),
      reactive: mean(rs, 'reactive_sla'),
      worstBelow: below.length ? Math.min(...below.map(r => r.improvement_pp)) : NaN,
      bestAbove: above.length ? Math.max(...above.map(r => r.improvement_pp)) : NaN,
    };
  });
  const both = per.filter(c => c.nb && c.na);
  return {
    per, both,
    meanBelow: both.reduce((s, c) => s + c.below, 0) / both.length,
    meanAbove: both.reduce((s, c) => s + c.above, 0) / both.length,
    drops: both.filter(c => c.above < c.below).length,
    worseThanReactive: both.filter(c => c.worstBelow < 0).length,
    worstNeg: Math.min(...rows.map(r => r.improvement_pp)),
    bestPos: Math.max(...rows.map(r => r.improvement_pp)),
    anyViolation: rows.some(r => r.pact_sla > 0 || r.reactive_sla > 0),
    reactiveViolates: rows.some(r => r.reactive_sla > 0),
    maxTicks: Math.round(Math.max(...rows.map(r => r.pact_sla)) / 100 * rows[0].n_runs * B.MAN.n_ticks_per_run),
    cellTicks: rows[0].n_runs * B.MAN.n_ticks_per_run,
    exactFrac: mean(rows, 'exact_frac'),
    nRuns: rows[0].n_runs,
  };
}
const CL5 = cliffStats(B.E5);
const CL = Object.fromEntries(REAL.map(w => [w, cliffStats(E18.filter(r => r.workload === w))]));

// ------------------------------------------------------------ E3 on recorded rows
const e3real = w => {
  const rs = E3.filter(r => r.workload === w);
  return { n: rs.length, dis: mean(rs, 'disagree_pct'), ea: mean(rs, 'exact_actions'),
           ra: mean(rs, 'rounded_actions'), dsla: mean(rs, 'delta_sla'), dover: mean(rs, 'delta_over') };
};
const e3syn = (() => {
  const rs = E3.filter(r => !REAL.includes(r.workload));
  return { dis: mean(rs, 'disagree_pct'), ea: mean(rs, 'exact_actions'), ra: mean(rs, 'rounded_actions') };
})();
const G3 = Object.fromEntries(REAL.map(w => [w, e3real(w)]));

// ------------------------------------------------------------ margin and ceiling
const agg = (rows, k) => ({ n: rows.length, moved: rows.filter(r => Number(r[k]) > 0).length,
                            max: rows.length ? Math.max(...rows.map(r => Number(r[k]))) : NaN,
                            total: rows.reduce((s, r) => s + Number(r[k]), 0) });
const fullReal = w => E4R.filter(r => r.variant === 'full' && r.workload === w);
const fullSyn = E4R.filter(r => r.variant === 'full' && !REAL.includes(r.workload));
const e1Arm = (m, real) => E1R.filter(r => r.method === m && (real ? REAL.includes(r.workload) : !REAL.includes(r.workload)));
const runs17 = B.csv('e17_trained_runs.csv');
const arm17 = (w, m) => runs17.filter(r => r.workload === w && r.method === m);
const e19 = (w, guard, sigma) => E19.find(r => r.workload === w && TRUTHY(r.safety_guard) === guard
  && Math.abs(r.forecast_noise_sigma - sigma) < 1e-9);
const e19ovr = w => E19.filter(r => r.workload === w && TRUTHY(r.safety_guard) && r.forecast_noise_sigma >= 0)
  .reduce((s, r) => s + Number(r.overrides_total), 0);
const e19sigmas = [...new Set(E19.filter(r => r.forecast_noise_sigma >= 0).map(r => r.forecast_noise_sigma))];
const e19curvesDiffer = w => e19sigmas.some(s => e19(w, true, s).sla_mean !== e19(w, false, s).sla_mean);

// verdict helpers
const lawVerdict = c => {
  if (!c.anyViolation) return 'not observable';
  if (!c.reactiveViolates) return `not testable (reactive never violates; ≤ ${c.maxTicks} ticks per cell)`;
  if (c.drops === c.both.length) return 'holds';
  if (c.drops === 0) return 'fails';
  return `partly (${c.drops} of ${c.both.length})`;
};
const reactVerdict = c => {
  if (!c.anyViolation) return 'not observable';
  if (!c.reactiveViolates) return 'not testable';
  return c.worseThanReactive ? `holds (${c.worseThanReactive} of ${c.both.length})` : 'fails';
};
const churnVerdict = g => (g.ra > g.ea ? 'holds' : 'fails');
const disVerdict = g => (g.dis > 0 ? 'holds' : 'fails');
const movedText = a => (a.moved ? `moved in ${a.moved} of ${a.n} runs (max ${f(a.max, 2)})` : `never moved (${a.n} runs)`);
const ovrText = a => (a.total ? `${a.total} overrides in ${a.moved} of ${a.n} runs` : `none in ${a.n} runs`);

// ------------------------------------------------------------ section
const section = [
  H2('Q.\u2003Recorded Traffic and a Trained Forecaster'),
  P('Every result up to this point was obtained on generated traces with bracketing predictors. This '
  + 'subsection repeats the central measurements on recorded request traffic and replaces the brackets '
  + 'with a forecaster trained on that traffic. It reports where the trained forecaster fell relative to '
  + 'the brackets, and which of the earlier findings survive the change of workload. A finding is said '
  + 'to survive only when the rule stated with Table XXI is met; findings that fail it are reported as failing.'),

  H3('The trace'),
  P(`The recorded workload is the 1998 FIFA World Cup web-site access log [11], days ${days[0]}–${days[1]} `
  + `(${dates[0]} to ${dates[1]}, Paris time), ${PROV.worldcup98.subset.n_files} log files and `
  + `${WC.total_requests.toLocaleString('en-US')} requests. The ten days are contiguous, so the chronological splits `
  + 'cut one unbroken series, and they span the start of the tournament: the first five are pre-tournament '
  + 'days and the opening match was played on 10 June, so the subset contains a diurnal cycle, a rise in '
  + 'base load and match-time bursts. The subset was fixed before any controller was run on it. Every file '
  + 'was decoded and checked against the reference '
  + 'counts shipped with the archive. Requests were binned into the 5 s control interval and divided by '
  + `5 to give an arrival rate; no smoothing or outlier removal was applied. The raw peak is `
  + `${f(WC.raw_peak_req_s, 1)} req/s (${paris(WC.raw_peak_bin)}), well beyond the ${f(WC.pool_capacity_req_s, 0)} req/s `
  + 'that the simulated pool can serve at its upper bound.'),
  P('That mismatch is handled by rescaling, which preserves shape but not absolute level. Two scalings '
  + 'are reported, because neither is satisfactory alone. The first, WC98/trace, multiplies the whole '
  + `series by ${f(WC.scale_factor, 4)} so that the trace peak is ${f(100 * WC.peak_fraction_of_capacity, 0)} % of pool capacity; `
  + `because the peak falls in the training split, the test split then peaks at only ${f(WC.test_peak_req_s, 1)} req/s `
  + 'and the pool is lightly loaded throughout. The second, WC98/test, multiplies the first by a further '
  + `${f(WCT.extra_scale, 3)} so that the test-split peak is ${f(100 * WCT.peak_fraction_of_capacity, 0)} % of capacity. The forecaster `
  + 'never sees this factor: its inputs and targets are standardised with training-split statistics, under '
  + 'which a constant multiple cancels, so the same frozen model serves both families and its output is '
  + 'multiplied back. Choosing a scale from the test peak does use one statistic of the test split, and '
  + 'this is stated as a limitation in Section VII.'),
  P(`The series is split chronologically ${f(100 * WC.train_fraction, 0)}/${f(100 * WC.val_fraction, 0)}/${f(100 * testFrac, 0)}: `
  + 'training on 5–10 June, validation on 11–12 June and test on 13–14 June. The test split is a Saturday '
  + 'and Sunday, so it is not a random sample of the week. Every closed-loop run on recorded traffic uses '
  + `a ${B.MAN.n_ticks_per_run}-tick window drawn from the test split, and the forecaster is checked at every tick to have `
  + 'been given only past observations. Fig. 18 shows the series and the splits. The trace also contains a '
  + 'brief dip at each local midnight, where the logging servers appear to have rotated their files; it is '
  + 'left in the data rather than repaired, and one test window contains it.'),
  P('Cutting windows from the middle of a recorded day exposed a defect in the harness that the synthetic '
  + 'traces had hidden. Every run used to open with four replicas. The synthetic generators begin near '
  + 'their base level, which four replicas can serve, but a recorded window begins at whatever load the '
  + `day has reached. In the first recorded runs, with that fixed start, the reactive controller on WC98/test `
  + `incurred ${e20('worldcup98_testpeak', 'fixed_4', 'reactive').violation_ticks} violation ticks over `
  + `${e20('worldcup98_testpeak', 'fixed_4', 'reactive').n_runs} windows, ${e20('worldcup98_testpeak', 'fixed_4', 'reactive')[EARLY]} of them `
  + `in the first ${EARLY_N} ticks, and the exact controller ${e20('worldcup98_testpeak', 'fixed_4', 'pact').violation_ticks}, `
  + `${e20('worldcup98_testpeak', 'fixed_4', 'pact')[EARLY]} of them early: the measured violations were the start-up transient, `
  + 'identical for every controller, and said nothing about control. The start state was therefore '
  + 'changed, for every workload alike, to four replicas or the requirement (2) of the first tick, '
  + `whichever is larger. Under that rule the counts are ${e20('worldcup98_testpeak', 'load_sized', 'reactive').violation_ticks} `
  + `(${e20('worldcup98_testpeak', 'load_sized', 'reactive')[EARLY]} early) and ${e20('worldcup98_testpeak', 'load_sized', 'pact').violation_ticks} `
  + `(${e20('worldcup98_testpeak', 'load_sized', 'pact')[EARLY]} early). Across the synthetic families the rule changes the `
  + `start of ${synChanged.changed} of ${synChanged.total} measured windows, which open slightly above what four replicas `
  + 'cover at the target. The change was made after this diagnosis and before any trained-arm result had '
  + 'been produced, and every experiment in this paper, synthetic and recorded, was re-run under it.'),
  img('fig_wc98_trace.png', 6.9),
  CAP(`Fig. 18. (a) The measured WorldCup98 arrival rate, days ${days[0]}–${days[1]}, at Δt = ${WC.dt_s} s before scaling, `
  + 'with the chronological training, validation and test splits shaded. (b) One test window as fed to the '
  + 'pool under each scaling, against pool capacity. Recorded traffic [11].'),

  H3('Forecast accuracy'),
  P(`The proposed forecaster is a dilated temporal convolutional network with ${TCN.n_params.toLocaleString('en-US')} parameters, `
  + `trained on the arrival rate with a ${TCN.window}-tick history, a ${TCN.horizon}-step horizon and the asymmetric loss of `
  + `the forecasting module (κ = ${TCN.kappa}), for ${TCN.epochs_run} epochs before early stopping on the validation split. `
  + 'Its prediction of arrivals is converted to the utilisation and memory series the controller consumes '
  + 'through the same capacity law used for the oracle. Four neural baselines with parameter counts within '
  + `${f(paramSpread, 1)} % of the TCN, an ARIMA model selected by AIC, and persistence were evaluated on the identical `
  + `${tcnH.n_windows.toLocaleString('en-US')} test origins. Table XIX reports the result and Fig. 19 the full per-horizon curves.`),
  mkTable(
    ['Model', 'Parameters', 'Epochs', 'nMAE h=1 (%)', `nMAE h=${Math.round(HMAX / 2)} (%)`,
     `nMAE h=${HMAX} (%)`, `MAE h=${HMAX}\n(req/s)`, `Bias h=${HMAX}\n(req/s)`, `Under-forecast\nh=${HMAX} (%)`],
    MODELS.filter(m => acc(m, 1)).map(m => [
      MNAME[m], String(acc(m, 1).n_params), acc(m, 1).epochs === '' ? '—' : String(acc(m, 1).epochs),
      f(acc(m, 1).nmae_pct, 2), f(acc(m, Math.round(HMAX / 2)).nmae_pct, 2), f(acc(m, HMAX).nmae_pct, 2),
      f(acc(m, HMAX).mae_req_s, 2), f(acc(m, HMAX).bias_req_s, 2), f(acc(m, HMAX).under_pct, 1),
    ]),
    [1700, 1100, 800, 1100, 1100, 1100, 1000, 1000, 1200], { left: false, size: 13 }),
  CAP(`Table XIX. Forecast error on the WorldCup98 test split, ${tcnH.n_windows.toLocaleString('en-US')} origins, one training seed `
  + 'per model. nMAE is mean absolute error as a percentage of the mean test arrival rate; bias is mean '
  + 'forecast minus actual; under-forecast is the share of forecasts below the actual value. Rates are in '
  + 'WC98/trace units.'),
  P(`The proposed model is not the most accurate. At the full horizon it ranks ${tcnRank} of ${atH.length}, with an nMAE of `
  + `${f(tcnH.nmae_pct, 2)} % against ${f(best.r.nmae_pct, 2)} % for ${MNAME[best.m]}, the best model, and `
  + `${f(perH.nmae_pct, 2)} % for persistence; it is more accurate than persistence at ${tcnBeatsPersist} of the ${nH} `
  + `horizons. At one step it is ${tcn1.nmae_pct < per1.nmae_pct ? 'more' : 'less'} accurate than persistence `
  + `(${f(tcn1.nmae_pct, 2)} % against ${f(per1.nmae_pct, 2)} %). The best neural model is ${MNAME[bestNeural.m]} at `
  + `${f(bestNeural.r.nmae_pct, 2)} %, and ARIMA, with ${ariH.n_params} parameters, is more accurate than the neural model `
  + `in ${ariWins} of the ${ariPairs} model–horizon pairs. This is consistent with the observation of Fog et al. [3] that `
  + 'classical models are competitive on smooth or strongly seasonal traffic.'),
  P(`The neural models share one further property: they are biased upward. The TCN over-forecasts by `
  + `${f(tcnH.bias_req_s, 2)} req/s on average at the full horizon and falls below the actual rate on only `
  + `${f(tcnH.under_pct, 1)} % of forecasts, against ${f(perH.under_pct, 1)} % for persistence and ${f(ariH.under_pct, 1)} % for ARIMA. `
  + `That is the intended effect of the asymmetric loss, which charges under-prediction ${TCN.kappa} times as heavily, and `
  + 'every neural model was trained with it, so the comparison among architectures is like for like. The '
  + 'comparison against ARIMA and persistence is not: those are fitted to be unbiased, and part of their '
  + 'accuracy advantage is the error the asymmetric loss accepts deliberately. The neural models were '
  + 'trained once each; no seed variance is reported, and the ranking among them should not be read as stable.'),
  img('fig_forecast_accuracy.png', 3.35),
  CAP('Fig. 19. Forecast error (nMAE) against forecast step for every model on the WorldCup98 test split. '
  + 'One training seed per neural model.'),

  H3('The trained forecaster against the brackets'),
  P(`The frozen TCN was then placed in the loop. Both controllers were run with each of the three `
  + `predictors on every disjoint ${B.MAN.n_ticks_per_run}-tick window of the test split, ${nWin('worldcup98')} windows per family, `
  + 'and the reactive baseline was run on the same windows. Table XX compares the trained arm with the two '
  + 'brackets for each metric, with paired Wilcoxon tests Holm-corrected across the whole table.'),
  mkTable(
    ['Family', 'Controller', 'Metric', 'Oracle', 'Trained', 'Persistence', 'Position',
     'p vs. oracle\n(Holm)', 'p vs. persist.\n(Holm)'],
    E17B.map(r => [FAM[r.workload], r.controller === 'pact' ? 'Exact' : 'Rounded', MET[r.metric],
                   f(r.oracle_mean, 2), f(r.trained_mean, 2), f(r.persist_mean, 2), POS[r.position],
                   pvH(r, 'oracle'), pvH(r, 'persist')]),
    [1100, 900, 1300, 900, 900, 1000, 1400, 1100, 1100], { left: false, size: 12 }),
  CAP(`Table XX. The trained TCN against both brackets on recorded traffic, ${nWin('worldcup98')} test windows per family. `
  + 'Lower is better for every metric. "Between" means the trained mean lies inside the interval spanned by '
  + 'the two bracket means. "1 (identical)" marks paired runs that were identical, which are excluded from the '
  + 'Holm correction.'),
  P(`The trained arm landed between the brackets on ${nBetween} of the ${E17B.length} comparisons, better than both on `
  + `${nBelow} and worse than both on ${nAbove}.`
  + (degenerate.length
    ? ` ${degenerate.length === 1 ? 'One' : degenerate.length} of the "between" comparisons ${degenerate.length === 1 ? 'is' : 'are'} degenerate, with all three `
      + `arms at the same value (${degenerate.map(describe).join('; ')}), so the trained arm sat inside a non-degenerate `
      + `bracket on ${nBetween - degenerate.length} of ${E17B.length - degenerate.length}.`
    : '')
  + (outside.length
    ? ' The comparisons outside the bracket are: '
      + outside.map(r => `${describe(r)} (${POS[r.position]}; trained ${f(r.trained_mean, 2)}, `
        + `oracle ${f(r.oracle_mean, 2)}, persistence ${f(r.persist_mean, 2)})`).join('; ') + '.'
    : '')
  + ` Against persistence, ${sigP.length} of the differences are significant after correction, `
  + `${sigPbetter.length} in the trained arm’s favour and ${sigPworse.length} against it; against the oracle, ${sigO.length} are.`),
  P('A result outside the bracket is not in itself evidence of an error, because Section V-M already showed '
  + 'that a predictor between the two extremes can beat both. It was nonetheless checked rather than '
  + 'assumed. Each forecast is computed from the history window ending at the current tick, the harness '
  + 'asserts at every tick that the observation it receives is the recorded value for that tick, every '
  + 'window starts after the validation split ends, and the checkpoint records the '
  + 'checksum of the trace it was trained on, which the harness verifies before use. None of these checks '
  + 'failed in any run. The outcome was reported as measured and the model was not re-tuned after it was seen.'),
  img('fig_trained_brackets.png', 6.9),
  CAP('Fig. 20. The trained arm against both brackets on recorded traffic: violation, over-provisioning and '
  + 'scaling actions, with 95 % bootstrap intervals over test windows. Table XX gives every metric.'),

  H3('Which findings survive'),
  P(`The horizon–dead-time sweep of Section V-A was repeated on each recorded family with the oracle `
  + `predictor, ${CL.worldcup98.nRuns} test windows per cell (Fig. 21). `
  + REAL.map(w => {
    const c = CL[w];
    if (!c.anyViolation) {
      return `On ${FAM[w]} no cell produced a single violation for either controller, so the transition cannot be observed.`;
    }
    if (!c.reactiveViolates) {
      const cells = E18.filter(r => r.workload === w && r.pact_sla > 0);
      return `On ${FAM[w]} the reactive baseline never violated in any cell, so the plant was not stressed enough to test the `
        + `transition. The predictive controller violated in ${cells.length} of the ${E18.filter(r => r.workload === w).length} cells, `
        + `${cells.every(r => TRUTHY(r.covers_coldstart)) ? 'all above the boundary' : 'some below the boundary'}, on at most ${c.maxTicks} of the `
        + `${c.cellTicks} ticks in a cell; read literally that is the opposite of the synthetic result, but it rests on a handful of ticks.`;
    }
    return `On ${FAM[w]}, averaged over the ${c.both.length} cold-start settings sampled on both sides of HΔt = τc, the `
      + `violation rate is ${f(c.meanBelow, 2)}\u202f% below the boundary and ${f(c.meanAbove, 2)}\u202f% above it, and it is lower above `
      + `the boundary at ${c.drops} of those ${c.both.length} settings. Below the boundary the predictive controller is worse than `
      + `reactive at ${c.worseThanReactive} of them, by up to ${f(Math.abs(Math.min(0, c.worstNeg)), 2)} points; above it, it is better by up to `
      + `${f(Math.max(0, c.bestPos), 2)} points.`;
  }).join(' ')),
  img('fig_horizon_law_real.png', 6.9),
  CAP(`Fig. 21. Violation rate over the cold-start × horizon grid on recorded traffic, one panel per scaling. `
  + `The line is HΔt = τc. Oracle predictor, ${CL.worldcup98.nRuns} test windows per cell.`),
  P('The integrality comparison of Section V-B was run on the recorded families alongside the synthetic '
  + 'ones. '
  + REAL.map(w => `On ${FAM[w]} the exact and rounded controllers disagree on ${f(G3[w].dis, 1)} % of ticks and the `
    + `rounded controller issues ${f(G3[w].ra, 1)} actions per run against ${f(G3[w].ea, 1)}, with a violation difference of `
    + `${f(G3[w].dsla, 2)} points.`).join(' ')
  + ` On the synthetic families the corresponding figures are ${f(e3syn.dis, 1)} %, ${f(e3syn.ra, 1)} and ${f(e3syn.ea, 1)}.`),
  P('The adaptive margin and the safety ceiling were examined in every recorded run. '
  + REAL.map(w => `On ${FAM[w]}, under the trained forecaster the margin ${movedText(agg(arm17(w, 'pact-trained'), 'gamma_max'))} `
    + `and under persistence it ${movedText(agg(arm17(w, 'pact-persist'), 'gamma_max'))}; the ceiling issued `
    + `${ovrText(agg(arm17(w, 'pact-trained'), 'overrides'))} under the trained forecaster and `
    + `${ovrText(agg(arm17(w, 'pact-oracle'), 'overrides'))} under the oracle.`).join(' ')
  + ' In the forecast-corruption sweep of Section V-F repeated on recorded traffic, the ceiling issued '
  + REAL.map(w => `${e19ovr(w)} overrides in total on ${FAM[w]}, and the curves with and without it `
    + `${e19curvesDiffer(w) ? 'differ' : 'coincide'}`).join(', and ') + '.'),
  mkTable(
    ['Finding', 'Synthetic', FAM.worldcup98, FAM.worldcup98_testpeak],
    [
      ['Violation falls across HΔt = τc (E5, E18)', lawVerdict(CL5), lawVerdict(CL.worldcup98), lawVerdict(CL.worldcup98_testpeak)],
      ['Short horizon worse than reactive (E5, E18)', reactVerdict(CL5), reactVerdict(CL.worldcup98), reactVerdict(CL.worldcup98_testpeak)],
      ['Exact and rounded decisions differ (E3)', disVerdict(e3syn), disVerdict(G3.worldcup98), disVerdict(G3.worldcup98_testpeak)],
      ['Rounding raises churn (E3)', churnVerdict(e3syn), churnVerdict(G3.worldcup98), churnVerdict(G3.worldcup98_testpeak)],
      ['Margin moves, oracle (E4 full)', movedText(agg(fullSyn, 'gamma_max')),
       movedText(agg(fullReal('worldcup98'), 'gamma_max')), movedText(agg(fullReal('worldcup98_testpeak'), 'gamma_max'))],
      ['Margin moves, persistence (E1)', movedText(agg(e1Arm('pact-persist', false), 'gamma_max')),
       movedText(agg(e1Arm('pact-persist', true).filter(r => r.workload === 'worldcup98'), 'gamma_max')),
       movedText(agg(e1Arm('pact-persist', true).filter(r => r.workload === 'worldcup98_testpeak'), 'gamma_max'))],
      ['Safety ceiling activates (E4 full)', ovrText(agg(fullSyn, 'overrides')),
       ovrText(agg(fullReal('worldcup98'), 'overrides')), ovrText(agg(fullReal('worldcup98_testpeak'), 'overrides'))],
    ],
    [3200, 2300, 2300, 2300], { left: true, size: 12 }),
  CAP('Table XXI. Findings re-tested on recorded traffic. A transition "holds" when violation is lower above '
  + 'the boundary than below it at every cold-start setting sampled on both sides; "short horizon worse" '
  + 'holds when the predictive controller is worse than reactive below the boundary at one or more of those '
  + 'settings, with the count given; "not observable" means no cell produced a violation. The margin and '
  + 'ceiling rows count runs, not verdicts, because the earlier claim that both were inert is itself corrected '
  + 'in Section V-E.'),
  H3('What moves the trained arm'),
  P('The TCN over-forecasts on average (Table XIX), so one candidate explanation for the trained arm’s position is '
  + 'that the bias alone acts as a capacity margin. This was tested directly rather than by re-tuning the model: '
  + `the oracle was shifted upward by the TCN’s mean bias over all horizons, ${f(E21[0].shift_req_s, 2)} req/s on ${FAM.worldcup98} `
  + `and ${f(e21('worldcup98_testpeak', 'pact', 'sla_violation_pct').shift_req_s, 2)} req/s on ${FAM.worldcup98_testpeak} after the same rescaling, and run on the same `
  + 'windows. Table XXII compares it with the unshifted oracle and the trained arm. For violation, over-provisioning, '
  + `cost and tail latency the shifted oracle moves in the same direction as the trained arm in ${capAgree} of `
  + `${capRows.length} comparisons, and its mean lies closer to the trained arm’s than the unshifted oracle’s does in `
  + `${capCloser} of them: the bias accounts for the capacity-side behaviour. For scaling actions and oscillations it `
  + `agrees in direction in only ${churnAgree} of ${churnRows.length}, and it does not reproduce the largest effect in Table XX, `
  + 'the rounded controller’s churn: '
  + REAL.map(w => `on ${FAM[w]} the shifted oracle gives ${f(e21(w, 'rounded', 'scale_actions').biased_oracle_mean, 1)} actions per run against `
    + `${f(e21(w, 'rounded', 'scale_actions').oracle_mean, 1)} unshifted and ${f(e21(w, 'rounded', 'scale_actions').trained_mean, 1)} trained`).join(', and ')
  + '. Some other property of the trained forecast damps the rounded controller. The smoothing effect measured '
  + 'in Section V-M is a candidate, but it was not tested here, and the paper does not claim it.'),
  mkTable(
    ['Family', 'Controller', 'Metric', 'Oracle', 'Oracle +\nTCN bias', 'Trained'],
    E21.filter(r => E21_SHOW.includes(r.metric)).map(r => [
      FAM[r.workload], r.controller === 'pact' ? 'Exact' : 'Rounded', MET[r.metric],
      f(r.oracle_mean, 2), f(r.biased_oracle_mean, 2), f(r.trained_mean, 2)]),
    [1500, 1200, 1700, 1200, 1300, 1200], { left: false, size: 12 }),
  CAP(`Table XXII. The oracle shifted by the TCN’s mean forecast bias, on the same ${E21[0].n_runs} test windows per family as `
  + 'Table XX. Three of the six metrics are shown; results/e21_bias_probe.csv has all six.'),
  ...(E22 ? e22Section() : []),
];

module.exports = { section, CL, CL5, G3, FAM, nBetween, nBelow, nAbove, outside, tcnRank, best, MNAME, atH,
                   degenerate, capAgree, capRows, churnAgree, churnRows, E22_EXISTS: E22 !== null,
                   E22_SLA_HELPS: E22 ? E22.filter(r => r.variant !== 'full' && r.metric === 'sla_violation_pct'
                     && isSigP(r.p_holm) && Number(r.delta) > 0) : [] };
