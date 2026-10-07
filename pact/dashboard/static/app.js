/* Live six-panel dashboard. No animation on data updates. */
const PACT = "#5b9fd4";
const TRUTH = "#8b93a0";
const WARN = "#d46a5b";
const GRID = "rgba(230, 237, 243, 0.12)";

const ticks = [];
let meta = { slo_ms: 200, dt: 5, horizon: 12, tau_c_s: null, gamma_max: 0.5 };
let plots = {};

function axis() {
  return {
    stroke: TRUTH,
    grid: { stroke: GRID },
    ticks: { stroke: GRID },
  };
}

function makePlot(el, series, extra = {}) {
  const opts = Object.assign(
    {
      width: el.clientWidth || 480,
      height: el.clientHeight || 180,
      legend: { show: true },
      scales: { x: { time: false } },
      axes: [axis(), axis()],
      series,
      cursor: { drag: { x: false, y: false } },
    },
    extra
  );
  return new uPlot(opts, [[], ...series.slice(1).map(() => [])], el);
}

function setSeries(plot, xs, ys) {
  plot.setData([xs, ...ys]);
}

function ingest(t) {
  ticks.push(t);
  if (ticks.length > 2000) ticks.shift();
  render();
}

function seriesX() {
  return ticks.map((t) => t.timestamp);
}

function renderReplicas() {
  const xs = seriesX();
  const required = ticks.map((t) => t.n_required);
  const predicted = ticks.map((t) => (t.demand && t.demand.length ? t.demand[0] : null));
  const actual = ticks.map((t) => t.observation.n);
  setSeries(plots.replicas, xs, [required, predicted, actual]);
}

function renderLatency() {
  const xs = seriesX();
  const p95 = ticks.map((t) => t.observation.ell_p95);
  const slo = ticks.map(() => meta.slo_ms);
  setSeries(plots.latency, xs, [p95, slo]);
}

function renderForecast() {
  if (!ticks.length) return;
  const last = ticks[ticks.length - 1];
  const fan = last.forecast || [];
  const H = fan.length || meta.horizon;
  const t0 = last.timestamp;
  const xs = ticks.map((t) => t.timestamp);
  xs.push(t0 + meta.dt, t0 + H * meta.dt);
  const actual = ticks.map((t) => t.observation.u);
  actual.push(null, null);
  const uNow = last.observation.u;
  const y1 = fan[0] ? fan[0][0] : null;
  const yH = fan.length ? fan[fan.length - 1][0] : null;
  const h1 = ticks.map(() => null);
  h1[h1.length - 1] = uNow;
  h1.push(y1, null);
  const hH = ticks.map(() => null);
  hH[hH.length - 1] = uNow;
  hH.push(null, yH);
  setSeries(plots.forecast, xs, [actual, h1, hH]);
}

function renderControl() {
  const xs = seriesX();
  const gamma = ticks.map((t) => t.drift.gamma);
  const ebar = ticks.map((t) => t.drift.ebar);
  const solve = ticks.map((t) => t.mpc_solve_time_s);
  setSeries(plots.control, xs, [gamma, ebar, solve]);
}

function gateFired(t) {
  const g = t.gate;
  if (!g) return false;
  return (
    g.rate_clipped !== g.rounded ||
    g.bound_clipped !== g.rate_clipped ||
    g.after_deadband !== g.bound_clipped ||
    g.cooldown_blocked
  );
}

function reason(t) {
  if (t.safety && t.safety.active) return "safety";
  if (t.gate && t.gate.cooldown_blocked) return "cooldown";
  if (t.gate && t.gate.after_deadband === 0 && t.gate.bound_clipped !== 0) return "deadband";
  if (t.mpc_u === null) return "direct";
  return "mpc";
}

function renderActions() {
  const body = document.getElementById("action-log");
  const rows = ticks.filter((t) => t.applied_u !== 0 || gateFired(t) || (t.safety && t.safety.active));
  body.replaceChildren();
  for (const t of rows.slice(-80)) {
    const g = t.gate || {};
    const tr = document.createElement("tr");
    if (t.safety && t.safety.active) tr.className = "override";
    tr.innerHTML = [
      t.tick,
      t.mpc_u ?? "—",
      g.rate_clipped ?? "—",
      g.bound_clipped ?? "—",
      g.after_deadband ?? "—",
      g.cooldown_blocked ? "yes" : "no",
      t.applied_u,
      reason(t),
    ]
      .map((v) => `<td>${v}</td>`)
      .join("");
    body.appendChild(tr);
  }
}

function renderTiles() {
  const n = ticks.length;
  if (!n) return;
  const sla = ticks.filter((t) => t.observation.ell_p95 > meta.slo_ms).length / n;
  let over = 0;
  for (const t of ticks) {
    const req = Math.max(t.n_required, 1);
    over += Math.max(t.n_desired - t.n_required, 0) / req;
  }
  over /= n;
  const actions = ticks.filter((t) => t.applied_u !== 0).length;
  const util = ticks.reduce((s, t) => s + t.observation.u, 0) / n;
  const ovr = ticks.filter((t) => t.safety && t.safety.active).length;
  document.getElementById("tile-sla").textContent = `${(sla * 100).toFixed(1)}%`;
  document.getElementById("tile-over").textContent = `${(over * 100).toFixed(1)}%`;
  document.getElementById("tile-actions").textContent = String(actions);
  document.getElementById("tile-util").textContent = util.toFixed(3);
  document.getElementById("tile-ovr").textContent = String(ovr);
}

function render() {
  if (!ticks.length || !plots.replicas) return;
  renderReplicas();
  renderLatency();
  renderForecast();
  renderControl();
  renderActions();
  renderTiles();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/ticks`);
  const el = document.getElementById("ws-state");
  ws.onopen = () => {
    el.textContent = "live";
  };
  ws.onclose = () => {
    el.textContent = "retrying";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    ingest(JSON.parse(ev.data));
  };
}

function boot() {
  plots.replicas = makePlot(document.getElementById("chart-replicas"), [
    { label: "t" },
    { label: "required", stroke: TRUTH, width: 1, paths: uPlot.paths.stepped({ align: 1 }) },
    { label: "Ñ(t+1)", stroke: PACT, width: 1, paths: uPlot.paths.stepped({ align: 1 }) },
    { label: "N(t)", stroke: "#c5d4e8", width: 1, paths: uPlot.paths.stepped({ align: 1 }) },
  ], {
    hooks: {
      draw: [
        (u) => {
          const tau = meta.tau_c_s;
          if (tau == null) return;
          u.ctx.fillStyle = "rgba(91, 159, 212, 0.12)";
          for (const t of ticks) {
            if (t.applied_u > 0) {
              const x0 = u.valToPos(t.timestamp, "x", true);
              const x1 = u.valToPos(t.timestamp + tau, "x", true);
              u.ctx.fillRect(
                x0,
                u.bbox.top,
                Math.max(1, x1 - x0),
                u.bbox.height
              );
            }
          }
        },
      ],
    },
  });
  plots.latency = makePlot(document.getElementById("chart-latency"), [
    { label: "t" },
    { label: "p95", stroke: PACT, width: 1 },
    { label: "SLO", stroke: WARN, width: 1, dash: [4, 4] },
  ], {
    hooks: {
      draw: [
        (u) => {
          const slo = meta.slo_ms;
          u.ctx.fillStyle = "rgba(212, 106, 91, 0.12)";
          for (const t of ticks) {
            if (t.observation.ell_p95 > slo) {
              const x0 = u.valToPos(t.timestamp, "x", true);
              const x1 = u.valToPos(t.timestamp + meta.dt, "x", true);
              u.ctx.fillRect(
                x0,
                u.bbox.top,
                Math.max(1, x1 - x0),
                u.bbox.height
              );
            }
          }
        },
      ],
    },
  });
  plots.forecast = makePlot(document.getElementById("chart-forecast"), [
    { label: "t" },
    { label: "actual u", stroke: TRUTH, width: 1 },
    { label: "h=1", stroke: PACT, width: 1 },
    { label: "h=H", stroke: "#7aa7d4", width: 1 },
  ]);
  plots.control = makePlot(document.getElementById("chart-control"), [
    { label: "t" },
    { label: "γ", stroke: PACT, width: 1, scale: "y" },
    { label: "ē", stroke: TRUTH, width: 1, scale: "y" },
    { label: "solve s", stroke: WARN, width: 1, scale: "y2" },
  ], {
    scales: { x: { time: false }, y: {}, y2: { side: 1 } },
    axes: [axis(), axis(), Object.assign({ side: 1, scale: "y2" }, axis())],
  });

  fetch("/api/meta")
    .then((r) => r.json())
    .then((m) => {
      meta = Object.assign(meta, m);
      document.getElementById("bind").textContent = m.bind || "127.0.0.1";
    })
    .catch(() => undefined);
  fetch("/api/ticks")
    .then((r) => r.json())
    .then((body) => {
      for (const t of body.ticks || []) ingest(t);
    })
    .catch(() => undefined);
  connect();
}

boot();
