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
const MAN = JSON.parse(fs.readFileSync(path.join(RES, 'manifest.json'), 'utf8'));

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
  return new Paragraph({
    alignment: o.align || AlignmentType.CENTER,
    spacing: { before: 70, after: 180 },
    children: [new TextRun({ text: t, size: 16, font: F })],
  });
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
  const w = Math.round(widthIn * 72);
  const h = Math.round(widthIn * 72 * dim.h / dim.w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 40 },
    children: [new ImageRun({ data, type: 'png',
                              transformation: { width: w, height: h } })],
  });
}
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

// ----------------------------------------------------------- tables
function mkTable(header, rows, widths, opts = {}) {
  const fsz = opts.size || 15;
  const cell = (txt, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: 'E9ECEF' } : undefined,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: String(txt).split('\n').map(line => new Paragraph({
      alignment: head ? AlignmentType.CENTER : (opts.left ? AlignmentType.LEFT : AlignmentType.CENTER),
      spacing: { after: 0, line: 210 },
      children: [new TextRun({ text: line, size: fsz, bold: !!head, font: F })],
    })),
  });
  return new Table({
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
      new TableRow({ tableHeader: true,
        children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map(r => new TableRow({
        children: r.map((c, i) => cell(c, widths[i], false)) })),
    ],
  });
}

module.exports = {
  d, P, RUNS, H1, H2, H3, CAP, BUL, EQ, M, MathRun, MathFraction, MathSum,
  MathRadical, sub, sup, img, mkTable, csv, E1, E2, E3, E4, E5, E6, E7, MAN,
  by, f, F,
};
