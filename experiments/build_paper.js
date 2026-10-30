/* Build the two-column IEEE-style paper from measured results.
 * Every numeric value is read from results/*.csv at build time.
 */
const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table,
  TableRow, TableCell, WidthType, ShadingType, BorderStyle, ImageRun,
  SectionType, Footer, PageNumber, LevelFormat, convertInchesToTwip,
  Math: M, MathRun, MathFraction, MathSubScript, MathSuperScript,
  MathSum, MathRadical,
} = d;

const ROOT = path.resolve(__dirname, '..');
const RES = path.join(ROOT, 'results');
const FIG = path.join(ROOT, 'figures');
const F = 'Times New Roman';

// ----------------------------------------------------------- csv loader
function csv(name) {
  const txt = fs.readFileSync(path.join(RES, name), 'utf8')
    .replace(/\r/g, '').trim();
  const [head, ...lines] = txt.split('\n');
  const keys = head.split(',');
  return lines.map(l => {
    // no quoted fields are produced by the writer
    const v = l.split(',');
    const o = {};
    keys.forEach((k, i) => {
      const raw = v[i];
      const n = Number(raw);
      o[k] = (raw !== '' && !Number.isNaN(n)) ? n : raw;
    });
    return o;
  });
}
const E1 = csv('e1_summary.csv');
const E2 = csv('e2_significance.csv');
const E3 = csv('e3_rounding_gap.csv');
const E4 = csv('e4_ablation_summary.csv');
const E5 = csv('e5_horizon_law.csv');
const E6 = csv('e6_frontier.csv');
const E7 = csv('e7_robustness.csv');
const E8 = csv('e8_sensitivity.csv');
const E9 = csv('e9_solvetime.csv');
const E10 = csv('e10_per_workload.csv');
const E11 = csv('e11_decisions.csv');
const E12 = csv('e12_coldstart.csv');
const MAN = JSON.parse(fs.readFileSync(path.join(RES, 'manifest.json'), 'utf8'));
const E15 = JSON.parse(fs.readFileSync(path.join(RES, 'e15_exact_vs_greedy.json'), 'utf8'));
const E8C = csv('e8_solver_compare.csv');
const E15T = csv('e15_ticks.csv');
// Fraction of the horizon sweep solved exactly (the rest fell back; see the
// tractability guard in pact/control/mpc_exact.py). Older result files have
// no such column, in which case this is reported as unknown.
const E5X = (() => {
  const rows = E5.filter(r => r.exact_frac !== undefined && r.exact_frac !== '');
  if (!rows.length) return null;
  const isExact = r => Number(r.exact_frac) >= 0.999;
  const exact = rows.filter(isExact).length;
  // Were the cells immediately either side of the H*dt = tau_c boundary
  // solved exactly? That is where the transition is measured.
  const taus = [...new Set(rows.map(r => Number(r.tau_c_s)))].sort((a, b) => a - b);
  let edgeExact = true;
  taus.forEach(tau => {
    const rs = rows.filter(r => Number(r.tau_c_s) === tau)
      .sort((a, b) => Number(a.H) - Number(b.H));
    const below = rs.filter(r => Number(r.H_dt) < tau);
    const above = rs.filter(r => Number(r.H_dt) >= tau);
    if (below.length && !isExact(below[below.length - 1])) edgeExact = false;
    if (above.length && !isExact(above[0])) edgeExact = false;
  });
  return { cells: rows.length, exact, fallback: rows.length - exact, edgeExact };
})();

const by = (rows, k, v) => rows.find(r => r[k] === v);
const f = (x, n = 2) => Number(x).toFixed(n);

// ----------------------------------------------------------- helpers
function P(text, o = {}) {
  const { size = 19, bold = false, italics = false,
          align = AlignmentType.JUSTIFIED, before = 0, after = 110 } = o;
  return new Paragraph({
    alignment: align, spacing: { before, after, line: 240 },
    children: [new TextRun({ text, size, bold, italics, font: F })],
  });
}
function RUNS(parts, o = {}) {
  const { align = AlignmentType.JUSTIFIED, before = 0, after = 110 } = o;
  return new Paragraph({
    alignment: align, spacing: { before, after, line: 240 },
    children: parts.map(p => new TextRun({
      text: p.t, size: p.size || 19, bold: !!p.b, italics: !!p.i, font: F,
    })),
  });
}
function H1(t) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text: t, size: 20, bold: true, font: F,
                             allCaps: true })],
  });
}
function H2(t) {
  return new Paragraph({
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: t, size: 19, bold: true, italics: true, font: F })],
  });
}
function H3(t) {
  return new Paragraph({
    spacing: { before: 150, after: 80 },
    children: [new TextRun({ text: t, size: 19, italics: true, font: F })],
  });
}
function CAP(t, o = {}) {
  const p = new Paragraph({
    alignment: o.align || AlignmentType.CENTER,
    spacing: { before: 70, after: 180 },
    keepLines: true,         // never split a caption across a column break
    children: [new TextRun({ text: t, size: 16, font: F })],
  });
  p.__cap = true;      // so a spanning table carries its caption with it
  return p;
}
function BUL(t) {
  return new Paragraph({
    numbering: { reference: 'b', level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 70, line: 240 },
    children: [new TextRun({ text: t, size: 19, font: F })],
  });
}

// ----------------------------------------------------------- equations (OMML)
function sub(base, s) {
  return new MathSubScript({
    children: [new MathRun(base)], subScript: [new MathRun(s)],
  });
}
function sup(base, s) {
  return new MathSuperScript({
    children: [new MathRun(base)], superScript: [new MathRun(s)],
  });
}
function EQ(children, num) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 110, after: 110 },
    children: [
      new M({ children }),
      new TextRun({ text: ` (${num})`, size: 19, font: F }),
    ],
  });
}

// ----------------------------------------------------------- figures
function img(file, widthIn) {
  const data = fs.readFileSync(path.join(FIG, file));
  const dim = pngSize(data);
  // Never let a figure exceed the frame it will sit in.
  const spans = widthIn * 1440 > COL_W;
  const maxIn = (spans ? TEXT_W : COL_W) / 1440;
  widthIn = Math.min(widthIn, maxIn);
  // docx sizes images in pixels at 96 dpi, not in points. Passing points
  // here rendered every figure at 72/96 = 75 % of its intended width.
  const w = Math.round(widthIn * PX_PER_IN);
  const h = Math.round(widthIn * PX_PER_IN * dim.h / dim.w);
  const par = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 40 },
    keepNext: true,          // keep the figure with its caption
    children: [new ImageRun({ data, type: 'png',
                              transformation: { width: w, height: h } })],
  });
  par.__spans = spans;
  par.__img = true;
  return par;
}
// Display equation rendered to an image by make_equations.py. OMML (see EQ
// below) stays in the source but is not what ships, because consumers that
// cannot render it drop the equation silently.
const PX_PER_IN = 96;   // docx image unit
const EQ_DPI = 600;
function EQI(num) {
  const file = `eq${String(num).padStart(2, '0')}.png`;
  const data = fs.readFileSync(path.join(FIG, file));
  const dim = pngSize(data);
  const natIn = dim.w / EQ_DPI;
  const spans = natIn > (COL_W / 1440) - 0.45;
  const maxIn = (spans ? TEXT_W : COL_W) / 1440 - 0.45;  // room for the number
  const wIn = Math.min(natIn, maxIn);
  const w = Math.round(wIn * PX_PER_IN);
  const h = Math.round(wIn * PX_PER_IN * dim.h / dim.w);
  const par = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    keepLines: true,
    children: [
      new ImageRun({ data, type: 'png', transformation: { width: w, height: h } }),
      new TextRun({ text: `\u2003(${num})`, size: 19, font: F }),
    ],
  });
  par.__spans = spans;
  return par;
}

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

// ----------------------------------------------------------- layout
// Page text width and the width of one body column, in twips. Everything
// that is laid out horizontally is fitted to one of these two numbers, so a
// table can never be wider than the frame that holds it.
const TEXT_W = 12240 - 1080 - 1080;          // 10080 twips = 7.00 in
const COL_GAP = 400;
const COL_W = Math.floor((TEXT_W - COL_GAP) / 2);  // 4840 twips = 3.36 in
const WIDE_COLS = 4;   // tables with this many columns span both columns

function fitW(widths, target) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total === target) return widths.slice();
  const scaled = widths.map(w => Math.max(260, Math.floor(w * target / total)));
  // put the rounding remainder on the widest column
  const drift = target - scaled.reduce((a, b) => a + b, 0);
  let widest = 0;
  scaled.forEach((w, i) => { if (w > scaled[widest]) widest = i; });
  scaled[widest] += drift;
  return scaled;
}

// ----------------------------------------------------------- tables
function mkTable(header, rows, widths, opts = {}) {
  // Wide tables span both body columns; narrow ones sit inside one.
  const spans = opts.span !== undefined
    ? opts.span
    : (widths.length >= WIDE_COLS || widths.reduce((a, b) => a + b, 0) > COL_W);
  const frame = spans ? TEXT_W : COL_W;
  widths = fitW(widths, frame);
  const fsz = opts.size || (spans ? 15 : 14);
  const cell = (txt, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: 'E9ECEF' } : undefined,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: String(txt).split('\n').map(line => new Paragraph({
      alignment: head ? AlignmentType.CENTER : (opts.left ? AlignmentType.LEFT : AlignmentType.CENTER),
      spacing: { after: 0, line: 210 },
      keepNext: true,        // hold the table together, and with its caption
      keepLines: true,
      children: [new TextRun({ text: line, size: fsz, bold: !!head, font: F })],
    })),
  });
  const t = new Table({
    alignment: AlignmentType.CENTER,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: '444444' },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: '444444' },
      left: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      right: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'AAAAAA' },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true,
        children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map(r => new TableRow({ cantSplit: true,
        children: r.map((c, i) => cell(c, widths[i], false)) })),
    ],
  });
  t.__spans = spans;   // read by the section assembler in paper.js
  return t;
}

function ALGO(title, lines, caption) {
  // Pseudocode lines run to ~65 monospace characters, which does not fit a
  // body column at a readable size, so these blocks span the text block.
  const body = [
    new Paragraph({
      spacing: { after: 50 },
      keepNext: true, keepLines: true,
      children: [new TextRun({ text: title, size: 16, bold: true, font: F })],
    }),
    ...lines.map(l => new Paragraph({
      spacing: { after: 0, line: 190 },
      keepNext: true, keepLines: true,
      children: [new TextRun({ text: l, size: 15, font: 'Consolas' })],
    })),
  ];
  if (caption) {
    body.push(new Paragraph({
      spacing: { before: 60, after: 0 },
      keepLines: true,
      children: [new TextRun({ text: caption, size: 14, font: F })],
    }));
  }
  const t = new Table({
    alignment: AlignmentType.CENTER,
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: [TEXT_W],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: '444444' },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: '444444' },
      left: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      right: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
    },
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: TEXT_W, type: WidthType.DXA },
      margins: { top: 90, bottom: 90, left: 90, right: 90 },
      children: body,
    })] })],
  });
  t.__spans = true;    // emitted in a one-column section by paper.js
  return t;
}

// A borderless one-cell table. Column balancing can move a table but cannot
// split a cantSplit row, so wrapping a figure with its caption in one of
// these stops the two being dealt into different columns.
function atomic(children, width) {
  return new Table({
    alignment: AlignmentType.CENTER,
    width: { size: width, type: WidthType.DXA },
    columnWidths: [width],
    borders: {
      top: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      bottom: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      left: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      right: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
    },
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: width, type: WidthType.DXA },
      margins: { top: 0, bottom: 0, left: 0, right: 0 },
      children,
    })] })],
  });
}

module.exports = {
  ALGO, atomic,
  d, P, RUNS, H1, H2, H3, CAP, BUL, EQ, EQI, M, MathRun, MathFraction, MathSum,
  MathRadical, sub, sup, img, mkTable, csv, E1, E2, E3, E4, E5, E6, E7, E8,
  E9, E10, E11, E12, E15, E15T, E5X, E8C, MAN, by, f, F, TEXT_W, COL_W, COL_GAP, fitW,
};
