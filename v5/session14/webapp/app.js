/* ============================================================
   SESSION 14 — Dense to Mixture-of-Experts — APP
   Renders window.SESSION_DATA (exported from results/*.json by
   moe_experiments/scripts/export_webapp_data.py) into tables and
   hand-rolled SVG charts. No build step, no libraries.
   ============================================================ */

const D = window.SESSION_DATA || { runs: [], curves: {}, matched: [], derived: {}, meta: {} };
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const SVGNS = 'http://www.w3.org/2000/svg';

// One fixed colour per run, used everywhere on the page.
const COL = { dense: '#4c8df0', moe_scratch: '#f0763f', dense_continue: '#8b7cf6', moe_upcycled: '#1fc48b',
              'abl_noise0.02': '#e87ba4', abl_no_lb_loss: '#7fd35a', abl_top1: '#ef4a4a', abl_experts16: '#f2b01e' };
const NAME = { dense: 'dense (pre-train)', moe_scratch: 'MoE from scratch', dense_continue: 'dense, continued',
               moe_upcycled: 'dense → MoE (upcycled)', 'abl_noise0.02': 'expert noise 0.02', abl_no_lb_loss: 'no aux loss',
               abl_top1: 'top-1 routing', abl_experts16: '16 experts' };
const RUN = Object.fromEntries(D.runs.map(r => [r.label, r]));
const CV = D.curves;
const DV = D.derived;

const fmt = {
  n: (v, d = 0) => v == null ? '–' : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }),
  s: (v, d = 3) => v == null ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d),
};

/* ═══════════════════════ Tooltip ═══════════════════════ */
const tip = $('#chart-tip');
function showTip(evt, html) { tip.innerHTML = html; tip.classList.add('show'); moveTip(evt); }
function moveTip(evt) { tip.style.left = (evt.clientX + 14) + 'px'; tip.style.top = (evt.clientY + 14) + 'px'; }
function hideTip() { tip.classList.remove('show'); }

/* ═══════════════════════ SVG helpers ═══════════════════════ */
function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, str, attrs = {}) {
  const t = svgEl('text', { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
function legend(el, items) {
  const div = document.createElement('div');
  div.className = 'chart-legend';
  div.innerHTML = items.map(i => `<span><i style="background:${i.color}"></i>${esc(i.name)}</span>`).join('');
  el.appendChild(div);
}
function hoverDot(g, x, y, color, html, r = 3.5) {
  const c = svgEl('circle', { cx: x, cy: y, r: r, fill: color, class: 'dot' }, g);
  const hit = svgEl('circle', { cx: x, cy: y, r: 9, fill: 'transparent', style: 'cursor:pointer' }, g);
  hit.addEventListener('mouseenter', e => { c.setAttribute('r', r + 2); showTip(e, html); });
  hit.addEventListener('mousemove', moveTip);
  hit.addEventListener('mouseleave', () => { c.setAttribute('r', r); hideTip(); });
}

/* ═══════════════════════ Line chart ═══════════════════════ */
function lineChart(el, o) {
  el.innerHTML = '';
  const W = Math.max(300, el.clientWidth || o.width || 620), H = o.height || 290, m = { t: 14, r: 18, b: 42, l: 56 };
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` }, el);
  const all = o.series.flatMap(s => s.x.map((x, i) => [x, s.y[i]])).filter(p => p[1] != null);
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  let x0 = o.xMin ?? Math.min(...xs), x1 = o.xMax ?? Math.max(...xs);
  let y0 = o.yMin ?? Math.min(...ys), y1 = o.yMax ?? Math.max(...ys);
  const tx = v => v, ty = o.yLog ? Math.log10 : (v => v);
  if (o.yLog) { y0 = Math.log10(y0); y1 = Math.log10(y1); }
  const X = v => m.l + (tx(v) - tx(x0)) / (tx(x1) - tx(x0) || 1) * (W - m.l - m.r);
  const Y = v => H - m.b - (ty(v) - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
  const grid = svgEl('g', { class: 'grid' }, svg), axis = svgEl('g', { class: 'axis' }, svg);
  const yt = o.yLog ? [10, 30, 100, 300, 1000, 3000, 10000, 30000].filter(v => Math.log10(v) >= y0 - 1e-9 && Math.log10(v) <= y1 + 1e-9)
                    : niceTicks(y0, y1, o.yTicks || 5);
  yt.forEach(v => {
    svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
    text(svg, m.l - 8, Y(v) + 4, (o.fmtY || (t => t))(v), { 'text-anchor': 'end' });
  });
  (o.xTicks || niceTicks(x0, x1, 6)).forEach(v => {
    svgEl('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid);
    text(svg, X(v), H - m.b + 16, (o.fmtX || (t => t))(v), { 'text-anchor': 'middle' });
  });
  svgEl('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, axis);
  if (o.xLabel) text(svg, (m.l + W - m.r) / 2, H - 6, o.xLabel, { 'text-anchor': 'middle', class: 'axis-label' });
  if (o.yLabel) text(svg, 14, (m.t + H - m.b) / 2, o.yLabel, { 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
  (o.refLines || []).forEach(r => {
    const col = r.color ? `stroke:${r.color}` : '';
    if (r.y != null) {
      svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), class: 'ref-line', style: col }, svg);
      text(svg, r.labelLeft ? m.l + 6 : W - m.r - 4, Y(r.y) + (r.labelBelow ? 14 : -6), r.label, { 'text-anchor': r.labelLeft ? 'start' : 'end' });
    } else {
      svgEl('line', { x1: X(r.x), x2: X(r.x), y1: m.t, y2: H - m.b, class: 'ref-line', style: col }, svg);
      text(svg, X(r.x) + 5, m.t + 12, r.label);
    }
  });
  const clipY = v => Math.max(m.t, Math.min(H - m.b, Y(v)));
  o.series.forEach(s => {
    const pts = s.x.map((x, i) => [x, s.y[i]]).filter(p => p[1] != null && p[0] >= x0 && p[0] <= x1);
    svgEl('path', {
      d: pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${clipY(p[1]).toFixed(1)}`).join(''),
      class: 'series', stroke: s.color, 'stroke-dasharray': s.dash || '', 'stroke-width': s.width || 2.2, opacity: s.opacity || 1,
    }, svg);
    if (s.dots !== false) {
      const g = svgEl('g', {}, svg);
      pts.forEach(p => {
        if (Y(p[1]) < m.t || Y(p[1]) > H - m.b) return;
        hoverDot(g, X(p[0]), Y(p[1]), s.color,
          `<b>${esc(s.name)}</b><br>${esc(o.xName || 'step')}: ${(o.fmtTipX || o.fmtX || (t => t))(p[0])}<br>${esc(o.yName || 'value')}: ${(o.fmtTipY || (t => (+t).toFixed(4)))(p[1])}`,
          s.dotSize || 3);
      });
    }
  });
  (o.markers || []).forEach(mk => {
    const g = svgEl('g', {}, svg), x = X(mk.x), y = clipY(mk.y), r = 6;
    svgEl('path', { d: `M${x - r},${y - r}L${x + r},${y + r}M${x + r},${y - r}L${x - r},${y + r}`, stroke: mk.color, 'stroke-width': 2.4 }, g);
    const hit = svgEl('circle', { cx: x, cy: y, r: 9, fill: 'transparent' }, g);
    hit.addEventListener('mouseenter', e => showTip(e, mk.tip)); hit.addEventListener('mousemove', moveTip); hit.addEventListener('mouseleave', hideTip);
  });
  (o.labels || []).forEach(l => text(svg, X(l.x), clipY(l.y), l.text, { 'text-anchor': l.anchor || 'start', class: 'value', style: l.color ? `fill:${l.color}` : '' }));
  if (o.legend !== false) legend(el, o.series.filter(s => s.name && !s.noLegend).map(s => ({ name: s.name, color: s.color })));
}

/* ═══════════════════════ Bar chart (non-negative) ═══════════════════════ */
function barChart(el, o) {
  el.innerHTML = '';
  const W = Math.max(280, el.clientWidth || 620), H = o.height || 270, m = { t: 24, r: 14, b: 46, l: 50 };
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` }, el);
  const n = o.cats.length;
  const yMax = o.yMax || Math.max(...o.values) * 1.18;
  const Y = v => H - m.b - v / yMax * (H - m.t - m.b);
  const grid = svgEl('g', { class: 'grid' }, svg);
  niceTicks(0, yMax, 5).forEach(v => {
    svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
    text(svg, m.l - 8, Y(v) + 4, (o.fmtY || (t => t))(v), { 'text-anchor': 'end' });
  });
  const band = (W - m.l - m.r) / n, bw = Math.min(54, band * 0.62);
  o.cats.forEach((cat, i) => {
    const cx = m.l + band * (i + 0.5), v = o.values[i];
    String(cat).split('\n').forEach((ln, j) => text(svg, cx, H - m.b + 16 + j * 13, ln, { 'text-anchor': 'middle' }));
    const rect = svgEl('rect', { x: cx - bw / 2, y: Y(v), width: bw, height: Math.max(0.5, Y(0) - Y(v)), rx: 3, fill: o.colors[i] }, svg);
    rect.addEventListener('mouseenter', e => showTip(e, `<b>${esc(String(cat).replace('\n', ' '))}</b><br>${(o.fmtTip || o.fmtVal || (t => t))(v)}`));
    rect.addEventListener('mousemove', moveTip); rect.addEventListener('mouseleave', hideTip);
    text(svg, cx, Y(v) - 5, (o.fmtVal || (t => t))(v), { 'text-anchor': 'middle', class: 'value' });
  });
  svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--border)' }, svg);
}

/* ═══════════════════════ Diverging horizontal bars ═══════════════════════ */
function divBars(el, rows, o = {}) {
  el.innerHTML = '';
  const W = Math.max(300, el.clientWidth || 620), rowH = 34, H = rows.length * rowH + 44, m = { t: 8, r: 16, b: 34, l: 150 };
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}` }, el);
  const ext = o.extent || Math.max(...rows.map(r => Math.abs(r.value))) * 1.35;
  const X = v => m.l + (v + ext) / (2 * ext) * (W - m.l - m.r);
  const grid = svgEl('g', { class: 'grid' }, svg);
  niceTicks(-ext, ext, 6).forEach(v => {
    svgEl('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid);
    text(svg, X(v), H - m.b + 16, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2), { 'text-anchor': 'middle' });
  });
  text(svg, (m.l + W - m.r) / 2, H - 4, o.xLabel || 'Δ val loss (nats)', { 'text-anchor': 'middle', class: 'axis-label' });
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + 5, x0 = X(0), x1 = X(r.value);
    text(svg, m.l - 10, y + rowH / 2 - 2, r.label, { 'text-anchor': 'end', style: 'fill:var(--text-secondary);font-size:12px' });
    const rect = svgEl('rect', { x: Math.min(x0, x1), y, width: Math.max(1, Math.abs(x1 - x0)), height: rowH - 12, rx: 3, fill: r.color }, svg);
    rect.addEventListener('mouseenter', e => showTip(e, `<b>${esc(r.label)}</b><br>${esc(r.tip || '')}`));
    rect.addEventListener('mousemove', moveTip); rect.addEventListener('mouseleave', hideTip);
    text(svg, r.value >= 0 ? x1 + 6 : x1 - 6, y + (rowH - 12) / 2 + 4, r.note, { 'text-anchor': r.value >= 0 ? 'start' : 'end', class: 'value' });
  });
  svgEl('line', { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: 'var(--text-muted)' }, svg);
}

/* ═══════════════════════ Heatmap ═══════════════════════ */
function heatmap(el, mat, o = {}) {
  el.innerHTML = '';
  const W = Math.max(320, el.clientWidth || 700), rows = mat.length, cols = mat[0].length;
  const m = { t: 8, r: 8, b: 34, l: 80 }, cw = Math.min(70, (W - m.l - m.r) / cols), ch = 46;
  const H = m.t + rows * ch + m.b;
  const svg = svgEl('svg', { viewBox: `0 0 ${m.l + cols * cw + m.r} ${H}` }, el);
  const vmax = o.vmax || 0.3, color = o.color || '#1fc48b';
  mat.forEach((row, i) => {
    text(svg, m.l - 10, m.t + i * ch + ch / 2 + 4, `MoE layer ${i}`, { 'text-anchor': 'end', style: 'fill:var(--text-secondary)' });
    row.forEach((v, j) => {
      const g = svgEl('g', {}, svg);
      const r = svgEl('rect', { x: m.l + j * cw + 1, y: m.t + i * ch + 1, width: cw - 2, height: ch - 2, rx: 4, fill: color, opacity: 0.08 + 0.92 * Math.min(1, v / vmax) }, g);
      text(g, m.l + j * cw + cw / 2, m.t + i * ch + ch / 2 + 4, v.toFixed(2), { 'text-anchor': 'middle', style: 'fill:var(--text-primary);font-weight:600;font-size:11.5px' });
      r.addEventListener('mouseenter', e => showTip(e, `<b>layer ${i} · expert ${j}</b><br>share of routed tokens: ${(v * 100).toFixed(1)}%<br>fair share 12.5%`));
      r.addEventListener('mousemove', moveTip); r.addEventListener('mouseleave', hideTip);
    });
  });
  for (let j = 0; j < cols; j++) text(svg, m.l + j * cw + cw / 2, H - m.b + 18, `E${j}`, { 'text-anchor': 'middle' });
}

/* ═══════════════════════ Donut ═══════════════════════ */
function donut(el, title, parts, center, sub, unit = 'M') {
  const wrap = document.createElement('div');
  wrap.className = 'donut-wrap card';
  wrap.style.padding = '18px';
  const svg = svgEl('svg', { viewBox: '0 0 120 120' });
  const total = parts.reduce((a, p) => a + p.value, 0);
  let a0 = -Math.PI / 2;
  parts.forEach(p => {
    const a1 = a0 + p.value / total * Math.PI * 2, large = a1 - a0 > Math.PI ? 1 : 0;
    const R = 54, r = 34, c = 60;
    const P = (a, rad) => `${c + rad * Math.cos(a)},${c + rad * Math.sin(a)}`;
    const path = svgEl('path', { d: `M${P(a0, R)}A${R},${R} 0 ${large} 1 ${P(a1, R)}L${P(a1, r)}A${r},${r} 0 ${large} 0 ${P(a0, r)}Z`, fill: p.color, stroke: 'var(--bg-card)', 'stroke-width': 1.5 }, svg);
    path.addEventListener('mouseenter', e => showTip(e, `<b>${esc(p.name)}</b><br>${p.value.toFixed(2)} ${unit} · ${(p.value / total * 100).toFixed(1)}%`));
    path.addEventListener('mousemove', moveTip); path.addEventListener('mouseleave', hideTip);
    a0 = a1;
  });
  text(svg, 60, 60, center, { 'text-anchor': 'middle', class: 'donut-center' });
  text(svg, 60, 75, sub, { 'text-anchor': 'middle', class: 'donut-sub' });
  wrap.appendChild(svg);
  const lg = document.createElement('div');
  lg.className = 'donut-legend';
  lg.innerHTML = `<strong style="color:var(--text-primary);font-size:13px">${esc(title)}</strong>` +
    parts.map(p => `<span><i style="background:${p.color}"></i>${esc(p.name)} — ${p.value.toFixed(2)} ${unit} (${(p.value / total * 100).toFixed(1)}%)</span>`).join('');
  wrap.appendChild(lg);
  el.appendChild(wrap);
}

const kv = (sel, rows) => { $(sel).innerHTML = rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join(''); };
const ser = (label, key, xKey, yKey, extra = {}) => ({ name: NAME[label], color: COL[label], x: CV[label][xKey], y: CV[label][yKey], ...extra });

/* ═══════════════════════ Hero + glance ═══════════════════════ */
function renderHero() {
  const up = RUN.moe_upcycled, dc = RUN.dense_continue, d = RUN.dense, ms = RUN.moe_scratch;
  const stats = [
    { val: `${d.valLoss.toFixed(3)} → ${up.valLoss.toFixed(3)}`, lbl: 'val loss: dense → upcycled MoE', cls: 'amber' },
    { val: '0.00000', lbl: 'Δ loss at conversion (lossless)', cls: 'indigo' },
    { val: fmt.s(DV.gapFinal), lbl: 'nats vs dense-continue control', cls: 'indigo' },
    { val: fmt.s(DV.scratchGap), lbl: 'MoE vs dense, same budget (from scratch)', cls: 'rose' },
    { val: `${d.paramsM.toFixed(0)}M → ${up.paramsM.toFixed(0)}M`, lbl: `stored params (${up.activeM.toFixed(1)}M active)` },
    { val: '−16%', lbl: 'steps to match the control (2,507 vs 3,000)' },
  ];
  $('#hero-stats').innerHTML = stats.map(s => `<div class="hero-stat"><span class="val ${s.cls || ''}">${esc(s.val)}</span><span class="lbl">${esc(s.lbl)}</span></div>`).join('');
}

function renderGlance() {
  const groups = [['dense', 'moe_scratch'], ['dense_continue', 'moe_upcycled', 'abl_noise0.02', 'abl_no_lb_loss', 'abl_top1', 'abl_experts16']];
  const bestOf = (g, key, dir = 'min') => (dir === 'min' ? Math.min : Math.max)(...g.map(l => RUN[l][key]));
  const rows = [];
  groups.forEach((g, gi) => {
    g.forEach(l => {
      const r = RUN[l];
      const b = k => Math.abs(r[k] - bestOf(g, k)) < 1e-9 ? 'best' : '';
      const bt = gi === 1 && Math.abs(r.tokSMedian - bestOf(g, 'tokSMedian', 'max')) < 1 ? 'best' : '';   // pre-training speeds are not comparable (disturbed dense run)
      const mv = r.maxVioEnd ? (r.maxVioEnd.reduce((a, c) => a + c, 0) / r.maxVioEnd.length) : null;
      rows.push(`<tr class="${l === 'moe_upcycled' ? 'hl' : ''}">
        <td class="name l"><i style="display:inline-block;width:9px;height:9px;border-radius:3px;background:${COL[l]};margin-right:7px"></i>${esc(NAME[l])}${l === 'moe_upcycled' ? ' 🏆' : ''}</td>
        <td>${gi === 0 ? 'pre-train' : 'continue'}</td>
        <td>${r.paramsM.toFixed(1)} / ${r.activeM.toFixed(1)}</td><td>${fmt.n(r.steps)}</td>
        <td>${r.valStart.toFixed(3)}</td><td class="${b('valLoss')}">${r.valLoss.toFixed(4)}</td><td>${r.valPpl.toFixed(1)}</td>
        <td>${r.trainLoss.toFixed(3)}</td><td>${r.wallMin.toFixed(1)}</td><td class="${bt}">${(r.tokSMedian / 1000).toFixed(0)}k</td>
        <td>${mv == null ? '–' : mv.toFixed(2)}</td><td>${r.deadEnd == null ? '–' : r.deadEnd}</td></tr>`);
    });
    if (gi === 0) rows.push('<tr><td colspan="12" style="text-align:left;color:var(--text-muted);font-size:11px;padding-top:12px">↓ continuation phase: all start from the dense checkpoint (val 3.627) and see the same batches</td></tr>');
  });
  $('#glance-table').innerHTML = `<thead><tr><th class="l">Run</th><th>Phase</th><th>Params total / active (M)</th><th>Steps</th><th>Start</th><th>Final val</th><th>Val ppl</th><th>Train</th><th>Wall min</th><th>Tok/s</th><th>MaxVio</th><th>Dead</th></tr></thead><tbody>${rows.join('')}</tbody>`;
}

/* ═══════════════════════ 2. Architecture ═══════════════════════ */
function renderArch() {
  const el = $('#arch-donuts'); el.innerHTML = '';
  donut(el, 'Dense model', [
    { name: 'token embedding (tied)', value: 25.73, color: '#4c8df0' }, { name: 'MLP (8 blocks)', value: 16.80, color: '#8b7cf6' },
    { name: 'attention', value: 8.40, color: '#1fc48b' }, { name: 'position embedding', value: 0.26, color: '#f2b01e' }, { name: 'layer norms', value: 0.02, color: '#8b93a7' }], '51.2M', 'parameters');
  donut(el, 'MoE model (8 experts, top-2, every 2nd block)', [
    { name: 'token embedding (tied)', value: 25.73, color: '#4c8df0' }, { name: 'experts (4 × 8 MLPs)', value: 67.19, color: '#f0763f' },
    { name: 'dense MLP (4 blocks)', value: 8.40, color: '#8b7cf6' }, { name: 'attention', value: 8.40, color: '#1fc48b' },
    { name: 'position embedding', value: 0.26, color: '#f2b01e' }, { name: 'routers + norms', value: 0.03, color: '#8b93a7' }], '110.0M', '59.6M active');
  $('#arch-table').innerHTML = `
    <thead><tr><th class="l">Component</th><th>Dense (M)</th><th>MoE (M)</th><th>Note</th></tr></thead><tbody>
    <tr><td class="name l">token embedding (tied head)</td><td>25.73</td><td>25.73</td><td class="l">50,257 × 512</td></tr>
    <tr><td class="name l">position embedding</td><td>0.26</td><td>0.26</td><td class="l">512 × 512</td></tr>
    <tr><td class="name l">attention</td><td>8.40</td><td>8.40</td><td class="l">8 blocks, unchanged by upcycling</td></tr>
    <tr><td class="name l">dense MLP</td><td>16.80</td><td>8.40</td><td class="l">4 of 8 blocks remain dense</td></tr>
    <tr><td class="name l">MoE experts</td><td>–</td><td>67.19</td><td class="l">4 blocks × 8 experts × 2.1M</td></tr>
    <tr><td class="name l">routers</td><td>–</td><td>0.016</td><td class="l">512 × 8 per MoE block</td></tr>
    <tr class="hl"><td class="name l"><strong>total stored</strong></td><td>51.21</td><td class="best">110.02</td><td class="l">2.15×</td></tr>
    <tr class="hl"><td class="name l"><strong>active per token</strong></td><td>51.21</td><td>59.63</td><td class="l">1.16× (top-2 of 8)</td></tr>
    <tr><td class="name l">FLOPs / token (MFLOPs)</td><td>330.9</td><td>381.4</td><td class="l">1.15×</td></tr></tbody>`;
}

/* ═══════════════════════ 3. Setup ═══════════════════════ */
function renderSetup() {
  kv('#kv-model', [['Architecture', 'pre-LN GPT, tied head'], ['Layers × heads × width', '8 × 8 × 512'], ['Context', '512 tokens'], ['MLP inner width', '2048 (4×), GELU'],
    ['Vocabulary', '50,257 (GPT-2 BPE)'], ['Dense parameters', '51.21M'], ['MoE', '8 experts, top-2, blocks 1/3/5/7'], ['MoE parameters', '110.02M (59.63M active)'], ['Precision', 'bf16 autocast, fp32 router']]);
  kv('#kv-data', [['Dataset', 'WikiText-103 (raw)'], ['Tokenizer', 'GPT-2 BPE (tiktoken)'], ['Train tokens', '119,085,169'], ['Validation tokens', '249,750'],
    ['Batches', 'random 513-token windows'], ['Eval batches', '50 × 32 × 512, fixed seed (identical for every run)'], ['Pre-training budget', '98.3M tokens = 0.83 epochs'], ['Continuation budget', '49.2M tokens']]);
  kv('#kv-train', [['Tokens / step', '32 × 512 = 16,384'], ['Optimiser', 'AdamW, β = (0.9, 0.95), wd 0.1 on matrices'], ['Grad clip', '1.0'],
    ['Pre-train LR', '6e-4, 300 warm-up, cosine → 6e-5'], ['Continue LR', '3e-4, 100 warm-up, cosine → 3e-5'], ['Balancing', 'bias γ 1e-3 · aux 1e-4 · z 1e-3'], ['Seed', '1337 (single seed)'], ['Tracking', 'Aim, 8 runs']]);
  lineChart($('#chart-lr'), {
    height: 250, xLabel: 'global step', yLabel: 'learning rate', yName: 'lr', fmtY: v => v.toExponential(0), fmtTipY: v => v.toExponential(2),
    series: [
      { name: 'pre-training (dense)', color: COL.dense, x: CV.dense.lrX, y: CV.dense.lr, dots: false },
      { name: 'continuation (both branches)', color: COL.moe_upcycled, x: CV.moe_upcycled.lrX, y: CV.moe_upcycled.lr, dots: false },
    ],
    refLines: [{ x: 6000, label: 'conversion', color: 'var(--text-muted)' }],
  });
}

/* ═══════════════════════ 4. Strategy ═══════════════════════ */
function renderStrategy() {
  const steps = [
    ['Load the trained dense model', 'Checkpoint at step 6,000 (val 3.627): the same weights seed both branches.'],
    ['Copy every second MLP into 8 experts', 'Blocks 1, 3, 5, 7: each expert is an exact copy of that block\'s dense MLP. Embeddings, attention, norms and blocks 0, 2, 4, 6 are copied unchanged.'],
    ['Add a small router', 'Linear(512 → 8), weights ~ N(0, 0.02), selection biases zero. Gates renormalised over the top-2, so the output equals the dense MLP\'s.'],
    ['Prove it is lossless', 'Loss before = loss after (|Δ| = 0), max logit difference 5.5e-5 (fp32).'],
    ['Continue with a fresh optimiser', 'LR re-warm 6e-5 → 3e-4 over 100 steps, cosine to 3e-5; loss-free balancing on; same batches as the dense control.'],
  ];
  $('#recipe-flow').innerHTML = steps.map((s, i) => `<div class="flow-step"><div class="flow-n">0${i + 1}</div><div class="flow-body"><h4>${s[0]}</h4><p>${s[1]}</p></div></div>`).join('');
  $('#why-table').innerHTML = `
    <thead><tr><th class="l">Run</th><th class="l">Init</th><th class="l">Answers</th></tr></thead><tbody>
    <tr><td class="name l">dense</td><td class="l">random</td><td class="l">part 1: the linear-layer baseline</td></tr>
    <tr><td class="name l">moe_scratch</td><td class="l">random</td><td class="l">part 2: MoE with the identical budget</td></tr>
    <tr class="hl"><td class="name l">moe_upcycled</td><td class="l">dense @ 6,000</td><td class="l">the trainer's task: convert and continue</td></tr>
    <tr><td class="name l">dense_continue</td><td class="l">dense @ 6,000</td><td class="l">control: is it better than just training longer?</td></tr>
    <tr><td class="name l">4 ablations</td><td class="l">dense @ 6,000</td><td class="l">which design choices matter?</td></tr></tbody>`;
}

/* ═══════════════════════ 5. Dense ═══════════════════════ */
function renderDense() {
  const c = CV.dense;
  lineChart($('#chart-dense-loss'), {
    xLabel: 'step', yLabel: 'loss (nats)', yName: 'loss', yMin: 3.4, yMax: 6.2, xMax: 6000,
    series: [
      { name: 'train (EMA)', color: COL.dense, x: c.stepX, y: c.lossEma, dots: false, opacity: 0.75, width: 1.8 },
      { name: 'validation', color: COL.dense_continue, x: c.evalStep, y: c.val, dash: '' },
    ],
    refLines: [{ y: 3.6269, label: 'final val 3.627', color: 'var(--text-muted)', labelLeft: false }],
  });
  lineChart($('#chart-dense-gn'), { xLabel: 'step', yLabel: 'grad-norm', yName: 'grad-norm', yMin: 0.4, yMax: 1.0, xMax: 6000,
    series: [{ name: 'dense', color: COL.dense, x: c.gnX, y: c.gnEma, dots: false }], legend: false });
}

/* ═══════════════════════ 6. Scratch ═══════════════════════ */
function renderScratch() {
  lineChart($('#chart-scratch-val'), {
    xLabel: 'step', yLabel: 'validation loss', yMin: 3.45, yMax: 5.4, xMax: 6000,
    series: [ser('dense', 'x', 'evalStep', 'val'), ser('moe_scratch', 'x', 'evalStep', 'val')],
    refLines: [{ y: 3.6269, label: 'dense final', color: 'var(--text-muted)', labelBelow: false }],
    markers: [{ x: DV.moeScratchCrossStep, y: 3.6269, color: COL.moe_scratch, tip: `MoE reaches dense's final loss at step ≈ ${fmt.n(DV.moeScratchCrossStep)}` }],
  });
  const d = CV.dense, m = CV.moe_scratch;
  const gap = m.val.map((v, i) => v - d.val[i]);
  const el = $('#chart-scratch-gap'); el.innerHTML = '';
  lineChart(el, { xLabel: 'step', yLabel: 'Δ val loss', yName: 'MoE − dense', xMax: 6000, yMin: -0.11, yMax: 0.01, fmtTipY: v => fmt.s(v, 4),
    series: [{ name: 'MoE − dense', color: COL.moe_scratch, x: d.evalStep.slice(1), y: gap.slice(1) }],
    refLines: [{ y: 0, label: 'equal', color: 'var(--text-muted)', labelLeft: true, labelBelow: true }], legend: false });
  $('#matched-table').innerHTML = `<thead><tr><th>Step</th><th>Tokens (M)</th><th>Dense val</th><th>MoE val</th><th>Gap</th><th>Perplexity dense → MoE</th></tr></thead><tbody>
    ${D.matched.map(r => `<tr class="${r.step === 6000 ? 'hl' : ''}"><td>${fmt.n(r.step)}</td><td>${(r.step * D.meta.tokPerStep / 1e6).toFixed(0)}</td><td>${r.dense.toFixed(4)}</td><td class="best">${r.moe.toFixed(4)}</td><td>${fmt.s(r.moe - r.dense, 4)}</td><td>${Math.exp(r.dense).toFixed(1)} → ${Math.exp(r.moe).toFixed(1)}</td></tr>`).join('')}</tbody>`;
  $('#cross-scratch').textContent = fmt.n(DV.moeScratchCrossStep);
}

/* ═══════════════════════ 7. Upcycle ═══════════════════════ */
let upView = 'full';
function renderUpcycle() {
  const cards = [
    { h: 'Lossless', big: '3.6269 → 3.6269', sub: `Δ loss 0.00000 · max logit diff 5.5e-5`, cls: 'win' },
    { h: 'It keeps learning', big: '3.627 → 3.502', sub: `perplexity 37.6 → 33.2 over 3,000 steps` },
    { h: 'vs the control', big: fmt.s(DV.gapFinal, 3) + ' nats', sub: `3.502 vs 3.521 · single seed` },
    { h: 'Steps to match control', big: '2,507 / 3,000', sub: 'crosses the control\'s final loss at step ≈ ' + fmt.n(DV.upCrossStep) },
  ];
  $('#up-stats').innerHTML = cards.map(c => `<div class="verdict-card ${c.cls || ''}"><h4>${c.h}</h4><div class="big">${c.big}</div><div class="muted">${c.sub}</div></div>`).join('');
  const el = $('#chart-up'), note = $('#up-note');
  if (upView === 'full') {
    lineChart(el, { height: 320, xLabel: 'global step', yLabel: 'validation loss', yMin: 3.4, yMax: 6.2,
      series: [ser('dense', 'x', 'evalStep', 'val'), ser('moe_scratch', 'x', 'evalStep', 'val', { dash: '5 4', opacity: 0.85 }), ser('dense_continue', 'x', 'evalStep', 'val'), ser('moe_upcycled', 'x', 'evalStep', 'val')],
      refLines: [{ x: 6000, label: 'conversion', color: 'var(--text-muted)' }] });
    note.textContent = 'Dashed: MoE trained from scratch (stops at step 6,000). After the fork the two continuations are almost on top of each other at this scale — see the zoom.';
  } else if (upView === 'zoom') {
    lineChart(el, { height: 320, xLabel: 'global step', yLabel: 'validation loss', yMin: 3.48, yMax: 3.76, xMin: 6000,
      series: [ser('dense_continue', 'x', 'evalStep', 'val'), ser('moe_upcycled', 'x', 'evalStep', 'val')],
      refLines: [{ y: 3.6269, label: 'loss at conversion (3.627)', color: 'var(--text-muted)', labelLeft: false }] });
    note.textContent = 'Both branches jump at step 6,250 (LR re-warm), recover through ≈ step 7,450 and then separate: the MoE ends 0.019 nats lower.';
  } else {
    const a = CV.moe_upcycled, b = CV.dense_continue;
    lineChart(el, { height: 320, xLabel: 'global step', yLabel: 'MoE − dense_continue', xMin: 6000, yMin: -0.025, yMax: 0.01, fmtTipY: v => fmt.s(v, 4),
      series: [{ name: 'moe_upcycled − dense_continue', color: COL.moe_upcycled, x: a.evalStep, y: a.val.map((v, i) => v - b.val[i]) }],
      refLines: [{ y: 0, label: 'equal', color: 'var(--text-muted)', labelLeft: true }], legend: false });
    note.textContent = 'Negative = the converted MoE is better. It is slightly worse for the first ≈ 800 steps, level at ≈ step 7,000 and pulls steadily ahead after that.';
  }
  const a = CV.moe_upcycled, b = CV.dense_continue;
  const take = (c, n) => ({ x: c.evalStep.slice(0, n), y: c.val.slice(0, n) });
  lineChart($('#chart-rw-lr'), { height: 230, xLabel: 'global step', yLabel: 'lr', fmtY: v => v.toExponential(0), fmtTipY: v => v.toExponential(2), xMin: 6000, xMax: 7000, xTicks: [6000, 6250, 6500, 6750, 7000],
    series: [{ name: 'lr', color: 'var(--text-muted)', x: a.lrX.filter(x => x <= 7000), y: a.lr.slice(0, a.lrX.filter(x => x <= 7000).length), dots: false }], legend: false,
    refLines: [{ y: 6e-5, label: 'where pre-training ended', color: 'var(--text-muted)', labelBelow: true }] });
  lineChart($('#chart-rw-val'), { height: 230, xLabel: 'global step', yLabel: 'val loss', xMin: 6000, xMax: 7500, xTicks: [6000, 6500, 7000, 7500], yMin: 3.6, yMax: 3.76,
    series: [{ name: NAME.dense_continue, color: COL.dense_continue, ...take(b, 7) }, { name: NAME.moe_upcycled, color: COL.moe_upcycled, ...take(a, 7) }],
    refLines: [{ y: 3.6269, label: '3.627', color: 'var(--text-muted)', labelLeft: true, labelBelow: true }] });
  lineChart($('#chart-rw-gn'), { height: 230, xLabel: 'global step', yLabel: 'grad-norm', xMin: 5000, yMin: 0.6, yMax: 1.0,
    series: [{ name: 'dense (pre-train)', color: COL.dense, x: CV.dense.gnX, y: CV.dense.gnEma, dots: false },
      { name: NAME.dense_continue, color: COL.dense_continue, x: b.gnX, y: b.gnEma, dots: false }, { name: NAME.moe_upcycled, color: COL.moe_upcycled, x: a.gnX, y: a.gnEma, dots: false }] });
  $('#bump-moe').textContent = fmt.s(DV.bumpMoe, 3); $('#bump-dense').textContent = fmt.s(DV.bumpDense, 3);
  $('#rec-moe').textContent = fmt.n(DV.recoverMoe); $('#rec-dense').textContent = fmt.n(DV.recoverDense);
  $('#gap-final').textContent = fmt.s(DV.gapFinal, 3); $('#cross-up').textContent = fmt.n(DV.upCrossStep);
}

/* ═══════════════════════ 8. Routing ═══════════════════════ */
let hmView = 'scratch';
function renderRouting() {
  const views = {
    scratch: [CV.moe_scratch.loadLast, 'MoE trained from scratch, end of training: every expert within 0.10–0.14 of the 0.125 fair share.'],
    up0: [CV.moe_upcycled.loadFirst, 'At conversion the freshly initialised router is arbitrary: layer 0 sends 29% of tokens to expert 2 and 27% to expert 7, 2% to experts 0 and 6. The experts are identical clones, so this costs no loss — but it is a poor start for balancing.'],
    up1: [CV.moe_upcycled.loadLast, 'After 3,000 steps the load has mostly evened out (0.09–0.18); layer 0 is the least balanced (MaxVio 0.45).'],
  };
  const [mat, note] = views[hmView];
  heatmap($('#chart-hm'), mat, { vmax: 0.3, color: hmView === 'scratch' ? COL.moe_scratch : COL.moe_upcycled });
  $('#hm-note').textContent = note;
  const dots = { dots: true, dotSize: 2.6 };
  lineChart($('#chart-maxvio'), { height: 250, xLabel: 'global step', yLabel: 'MaxVio', yMin: 0, yMax: 1.5,
    series: [ser('moe_scratch', 'x', 'evalStep', 'maxVioMean', dots), ser('moe_upcycled', 'x', 'evalStep', 'maxVioMean', dots)] });
  lineChart($('#chart-entropy'), { height: 250, xLabel: 'global step', yLabel: 'normalised entropy', yMin: 0.84, yMax: 1.005, fmtY: v => v.toFixed(2),
    series: [ser('moe_scratch', 'x', 'evalStep', 'entropyMean', dots), ser('moe_upcycled', 'x', 'evalStep', 'entropyMean', dots)] });
  lineChart($('#chart-dead'), { height: 250, xLabel: 'global step', yLabel: 'dead experts', yMin: 0, yMax: 5.5, fmtTipY: v => v,
    series: [ser('moe_upcycled', 'x', 'evalStep', 'dead', dots), ser('abl_experts16', 'x', 'evalStep', 'dead', dots), ser('abl_top1', 'x', 'evalStep', 'dead', dots)] });
}

/* ═══════════════════════ 9. Experts ═══════════════════════ */
function renderExperts() {
  const dv = [[0, 0.9284, 0.9232, 0.289], [1, 0.9303, 0.9260, 0.283], [2, 0.9261, 0.9231, 0.288], [3, 0.9334, 0.9289, 0.284]];
  $('#div-table').innerHTML = `<thead><tr><th>MoE layer</th><th>Mean cosine sim</th><th>Min cosine sim</th><th>Mean drift</th></tr></thead><tbody>
    <tr><td>at conversion</td><td>1.0000</td><td>1.0000</td><td>0</td></tr>
    ${dv.map(r => `<tr><td>${r[0]}</td><td>${r[1].toFixed(4)}</td><td>${r[2].toFixed(4)}</td><td>${(r[3] * 100).toFixed(1)}%</td></tr>`).join('')}</tbody>`;
  const T = (list) => list.map(([t, l]) => `<span class="tok">${esc(t)}</span>×${l}`).join(' ');
  const rows = [
    ['0', '6', '3.4%', [['…', 28.6], ['critics', 21.6], ['"', 19.6], ['others', 19.3], ['!', 12.2]]],
    ['0', '7', '2.0%', [['iversary', 49.7], ['wik', 49.7], ['intel', 49.7], ['ZIP', 49.7], ['nz', 49.7]]],
    ['0', '5', '10.1%', [['@', 8.7], ['did', 7.7], ['She', 7.1], ['performance', 6.7], ['video', 6.4]]],
    ['1', '5', '15.6%', [['=', 6.3], ['This', 5.8], ['The', 3.9], ['<|endoftext|>', 3.4]]],
    ['1', '6', '11.1%', [['including', 8.2], ['included', 8.0], ['while', 7.0], ['After', 7.0], ['many', 6.9]]],
    ['1', '3', '10.6%', [['were', 6.9], ['I', 6.3], ['they', 6.2], ['have', 6.2], ['not', 6.0]]],
  ];
  $('#expert-table').innerHTML = `<thead><tr><th>Layer</th><th>Expert</th><th>Share</th><th class="l">Tokens (lift)</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="l" style="font-family:Inter,sans-serif">${T(r[3])}</td></tr>`).join('')}</tbody>`;
}

/* ═══════════════════════ 10. Ablations ═══════════════════════ */
function renderAblate() {
  const ctl = RUN.dense_continue.valLoss;
  const order = ['abl_experts16', 'abl_noise0.02', 'abl_no_lb_loss', 'moe_upcycled', 'dense_continue', 'abl_top1'];
  const rows = order.map(l => ({ label: l === 'moe_upcycled' ? 'upcycled (reference)' : NAME[l], value: RUN[l].valLoss - ctl, color: COL[l],
    note: `${fmt.s(RUN[l].valLoss - ctl, 4)} (${RUN[l].valLoss.toFixed(4)})`, tip: `final val loss ${RUN[l].valLoss.toFixed(4)}` }));
  divBars($('#chart-abl-bars'), rows, { extent: 0.075 });
  const base = CV.dense_continue.val;
  lineChart($('#chart-abl-gap'), { height: 270, xLabel: 'global step', yLabel: 'Δ vs dense_continue', xMin: 6000, yMin: -0.03, yMax: 0.03, fmtTipY: v => fmt.s(v, 4),
    series: ['abl_experts16', 'abl_noise0.02', 'abl_no_lb_loss', 'moe_upcycled', 'abl_top1'].map(l => ({ name: l === 'moe_upcycled' ? 'upcycled (reference)' : NAME[l], color: COL[l], x: CV[l].evalStep, y: CV[l].val.map((v, i) => v - base[i]), dotSize: 2.4 })),
    refLines: [{ y: 0, label: 'dense_continue', color: 'var(--text-muted)', labelLeft: true, labelBelow: true }] });
  const vio = l => RUN[l].maxVioEnd ? RUN[l].maxVioEnd.map(v => v.toFixed(2)).join(' / ') : '–';
  const order2 = ['abl_experts16', 'abl_noise0.02', 'abl_no_lb_loss', 'moe_upcycled', 'dense_continue', 'abl_top1'];
  const best = Math.min(...order2.map(l => RUN[l].valLoss));
  $('#abl-table').innerHTML = `<thead><tr><th class="l">Run</th><th class="l">Change</th><th>Params (M)</th><th>Final val</th><th>Δ vs control</th><th>MaxVio (L0–L3)</th><th>Dead</th><th>Tok/s</th></tr></thead><tbody>
    ${order2.map(l => {
      const ch = { abl_experts16: '16 experts, top-2', 'abl_noise0.02': 'noise 0.02 × weight-std on each copy', abl_no_lb_loss: 'aux loss 0 (bias balancing still on)', moe_upcycled: 'plain copies, 8 experts, top-2', dense_continue: 'no conversion', abl_top1: 'top-1 routing' }[l];
      return `<tr class="${l === 'moe_upcycled' ? 'hl' : ''}"><td class="name l"><i style="display:inline-block;width:9px;height:9px;border-radius:3px;background:${COL[l]};margin-right:7px"></i>${l === 'moe_upcycled' ? 'upcycled (reference)' : NAME[l]}</td><td class="l">${ch}</td>
        <td>${RUN[l].paramsM.toFixed(0)}</td><td class="${RUN[l].valLoss === best ? 'best' : (l === 'abl_top1' ? 'worst' : '')}">${RUN[l].valLoss.toFixed(4)}</td><td>${fmt.s(RUN[l].valLoss - ctl, 4)}</td>
        <td class="${l === 'abl_top1' ? 'worst' : ''}">${vio(l)}</td><td class="${l === 'abl_top1' ? 'worst' : ''}">${RUN[l].deadEnd ?? '–'}</td><td>${(RUN[l].tokSMedian / 1000).toFixed(0)}k</td></tr>`;
    }).join('')}</tbody>`;
}

/* ═══════════════════════ 11. Cost ═══════════════════════ */
function renderCost() {
  const l = ['dense_continue', 'moe_upcycled', 'abl_top1', 'abl_experts16'];
  barChart($('#chart-tok'), { height: 260, cats: ['dense', 'MoE\ntop-2', 'MoE\ntop-1', 'MoE\n16 experts'], values: l.map(k => RUN[k].tokSMedian / 1000), colors: l.map(k => COL[k]), fmtVal: v => v.toFixed(0) + 'k', yMax: 140, fmtTip: v => v.toFixed(1) + 'k tokens/s' });
  const sps = RUN.moe_upcycled.wallMin / 3000;
  const mins = [RUN.dense_continue.wallMin, RUN.moe_upcycled.wallMin, sps * 2507];
  barChart($('#chart-min'), { height: 260, cats: ['dense\n3,000 steps', 'MoE\n3,000 steps', 'MoE\n2,507 steps'], values: mins, colors: [COL.dense_continue, COL.moe_upcycled, COL.moe_upcycled], fmtVal: v => v.toFixed(1), yMax: 14, fmtTip: v => v.toFixed(2) + ' min' });
}

/* ═══════════════════════ 13. Conclusions ═══════════════════════ */
function renderConclusions() {
  const findings = [
    '<strong>The conversion is exactly lossless</strong> (loss 3.6269 before and after; logits within 5.5e-5), because experts are copies and gates are renormalised.',
    '<strong>The converted model keeps training and reduces loss</strong>: 3.627 → 3.502 in 3,000 steps (perplexity 37.6 → 33.2).',
    '<strong>It beats continuing the dense model</strong> by 0.019 nats on identical batches, and matches the control\'s final loss in 2,507 instead of 3,000 steps.',
    '<strong>From scratch, the MoE leads dense by 0.096 nats</strong> at the same steps, reaching dense\'s final loss ≈ 25% earlier.',
    '<strong>Loss-free balancing works</strong>: from scratch MaxVio 0.03–0.15 and no dead experts; the 1e-4 auxiliary loss is not needed.',
    '<strong>The second expert is essential</strong>: top-1 routing starves the router of gradient and ends 0.019 <em>worse</em> than the dense control, with 5 dead experts.',
  ];
  const caveats = [
    '<strong>One seed per run.</strong> The 0.019 gap is steady over the last ~1,500 steps but has no error bar; ablation gaps of ≈ 0.002 are within single-seed noise.',
    '<strong>Not compute- or time-matched.</strong> The MoE uses 2.15× the stored and ≈ 1.15× the per-token FLOPs, and ran at ≈ 0.71× the dense tokens/s (Python expert loop). At equal time the dense model is ahead.',
    '<strong>Both continuations bump +0.11</strong> when the LR re-warms; a gentler re-warm was not tried.',
    '<strong>Peak memory was not captured</strong> (the trainer queried the default CUDA device while the run used <code>cuda:1</code>); fixed in code, not re-measured.',
    '<strong>Throughput during pre-training was noisy</strong> (median 60k vs mean 83k tokens/s, likely a shared GPU).',
    '<strong>Next:</strong> 2–3 seeds; an iso-time comparison with grouped-GEMM experts; a gentler LR re-warm; the implemented-but-unrun exploration window, sigmoid scores and drop-upcycling; a from-scratch MoE trained for the full 9,000 steps.',
  ];
  $('#conc-findings').innerHTML = findings.map(f => `<li>${f}</li>`).join('');
  $('#conc-caveats').innerHTML = caveats.map(f => `<li>${f}</li>`).join('');
  const ok = '<span class="badge badge-green">pass</span>', info = '<span class="badge badge-blue">info</span>';
  $('#check-table').innerHTML = `<thead><tr><th class="l">Check</th><th class="l">Evidence</th><th>Status</th></tr></thead><tbody>
    <tr><td class="name l">Conversion is lossless</td><td class="l">3.6269 → 3.6269, |Δ| 0.00000</td><td>${ok}</td></tr>
    <tr><td class="name l">MoE keeps training after conversion</td><td class="l">3.627 → 3.502 over 3,000 steps (every eval below the last from step 6,250)</td><td>${ok}</td></tr>
    <tr><td class="name l">MoE beats the dense-continue control</td><td class="l">3.502 vs 3.521 (−0.019), identical batches, single seed</td><td>${ok}</td></tr>
    <tr><td class="name l">No dead experts at the end</td><td class="l">0 dead; transient 0 → 2 → 0 in steps 6,250–7,000</td><td>${ok}</td></tr>
    <tr><td class="name l">Experts diverged from clones</td><td class="l">cosine sim 0.926–0.933, drift 28–29%</td><td>${ok}</td></tr>
    <tr><td class="name l">MoE from scratch vs dense, same budget</td><td class="l">3.531 vs 3.627 (−0.096)</td><td>${info}</td></tr>
    <tr><td class="name l">Compute- / wall-clock-matched win</td><td class="l">not shown: dense is faster per second here</td><td><span class="badge badge-yellow">not shown</span></td></tr></tbody>`;
}

function renderRepro() {
  kv('#kv-artifacts', [['Per-run histories', 'results/*.json (8 runs)'], ['Figures', 'assets/*.png (scripts/make_figures.py)'], ['Aim repo', '.aim/ · experiments session14, session14_ablations'],
    ['Notebooks', 'notebooks/01…04 (executed)'], ['Checkpoints', 'checkpoints/*.pt (not committed)'], ['This page\'s data', 'webapp/data.js (scripts/export_webapp_data.py)']]);
}

/* ═══════════════════════ Navigation / tabs ═══════════════════════ */
function initNav() {
  const items = $$('.nav-item');
  const sections = items.map(a => document.getElementById(a.dataset.target)).filter(Boolean);
  const onScroll = () => {
    const y = window.scrollY + 120;
    let current = sections[0];
    sections.forEach(s => { if (s.offsetTop <= y) current = s; });
    items.forEach(a => a.classList.toggle('active', a.dataset.target === current.id));
    const h = document.documentElement;
    $('#reading-progress').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%';
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }), { threshold: 0.08 });
  $$('.card, .fix-card, .verdict-card, .callout').forEach(c => { c.classList.add('reveal-item'); obs.observe(c); });
}
function tabs(ids, onPick) {
  ids.forEach(id => $('#' + id).addEventListener('click', () => {
    ids.forEach(o => $('#' + o).classList.toggle('active', o === id)); onPick(id);
  }));
}
function initTabs() {
  tabs(['up-full', 'up-zoom', 'up-gap'], id => { upView = id.replace('up-', ''); renderUpcycle(); });
  tabs(['hm-scratch', 'hm-up0', 'hm-up1'], id => { hmView = id.replace('hm-', ''); renderRouting(); });
  const lb = $('#lightbox');
  $$('.gallery img').forEach(img => img.addEventListener('click', () => { $('#lightbox-img').src = img.src; lb.classList.add('show'); }));
  lb.addEventListener('click', () => lb.classList.remove('show'));
}

const CHARTS = [renderSetup, renderDense, renderScratch, renderUpcycle, renderRouting, renderAblate, renderCost];
document.addEventListener('DOMContentLoaded', () => {
  renderHero(); renderGlance(); renderArch(); renderStrategy(); renderExperts(); renderConclusions(); renderRepro();
  CHARTS.forEach(f => f());
  initTabs(); initNav();
  let timer, lastW = window.innerWidth;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; CHARTS.forEach(f => f()); } }, 150);
  });
});
