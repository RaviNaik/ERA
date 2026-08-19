/* ============================================================
   SESSION 7 — Fourier Embeddings — APP
   ============================================================ */

const D = window.SESSION_DATA || {};
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/* ═══════════════════════════════════════════════════════════
   Theory verdict grid (theory-only, §9)
   ═══════════════════════════════════════════════════════════ */
function renderTheoryVerdict() {
  const grid = $('#theory-verdict-grid');
  if (!grid || !D.verdictTheory) return;
  const groups = [
    { key: 'solved', cls: 'solved', label: 'Solved', items: D.verdictTheory.solved },
    { key: 'unsolved', cls: 'unsolved', label: 'Not Solved', items: D.verdictTheory.unsolved },
    { key: 'new', cls: 'new-issue', label: 'New Issue Introduced', items: D.verdictTheory.newIssues },
  ];
  groups.forEach(g => {
    g.items.forEach(item => {
      const card = el('div', `verdict-card ${g.cls}`);
      card.innerHTML = `
        <span class="vc-status">${g.label}</span>
        <div class="vc-title">${item.title}</div>
        <div class="vc-desc">${item.desc}</div>
        <div class="vc-ref">${item.ref}</div>
      `;
      grid.appendChild(card);
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   Parameter accounting — theory table (§4.5)
   ═══════════════════════════════════════════════════════════ */
function renderParamTheoryTable() {
  const tbody = $('#param-theory-tbody');
  if (!tbody || !D.paramTheoryV5) return;
  D.paramTheoryV5.forEach(row => {
    const tr = el('tr', '', `
      <td>${row.codec}</td>
      <td class="mono">${row.D}</td>
      <td class="mono">${row.params}</td>
      <td>${row.wall}</td>
      <td>${row.decoupled}</td>
    `);
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   Experimental setup kv-grid (§11)
   ═══════════════════════════════════════════════════════════ */
function renderSetup() {
  const grid = $('#setup-kv-grid');
  if (!grid || !D.setupRows) return;
  D.setupRows.forEach(row => {
    const item = el('div', 'kv-item', `
      <span class="kv-key">${row.k}</span>
      <span class="kv-val mono">${row.v}</span>
    `);
    grid.appendChild(item);
  });
}

/* ═══════════════════════════════════════════════════════════
   Parameter efficiency, measured (§13)
   ═══════════════════════════════════════════════════════════ */
function renderParamMeasured() {
  const tbody = $('#param-measured-tbody');
  if (!tbody || !D.paramMeasured) return;
  D.paramMeasured.forEach(row => {
    const tr = el('tr', '', `
      <td>${row.arm}</td>
      <td class="mono">${row.total}</td>
      <td class="mono">${row.codec}</td>
      <td>${row.share}</td>
      <td>${row.reduction}</td>
    `);
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   Final performance comparison (§15)
   ═══════════════════════════════════════════════════════════ */
function renderFinalPerf() {
  const tbody = $('#final-perf-tbody');
  if (!tbody || !D.finalPerf) return;
  D.finalPerf.forEach(row => {
    const lossCls = row.best ? 'best-val' : (row.worst ? 'worst-val' : '');
    const tr = el('tr', '', `
      <td>${row.arm}</td>
      <td class="mono ${lossCls}">${row.finalLoss.toFixed(4)}</td>
      <td class="mono">${row.bestLoss.toFixed(4)}</td>
      <td class="mono">${row.ppl.toFixed(4)}</td>
      <td class="mono">${row.wall}</td>
    `);
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   Collision analysis table (§16)
   ═══════════════════════════════════════════════════════════ */
function renderCollisions() {
  const tbody = $('#collision-tbody');
  if (!tbody || !D.collisionRows) return;
  D.collisionRows.forEach(row => {
    const tr = el('tr', '', `
      <td>${row.script}</td>
      <td class="mono">${row.kronecker}</td>
      <td class="mono">${row.fourier}</td>
    `);
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   Order-sensitivity probe (§17)
   ═══════════════════════════════════════════════════════════ */
function renderOrderSensitivity() {
  const tbody = $('#order-summary-tbody');
  if (tbody && D.orderSensitivitySummary) {
    D.orderSensitivitySummary.forEach(row => {
      const tr = el('tr', '', `
        <td>${row.metric}</td>
        <td class="mono ${row.best === 'kronecker' ? 'best-val' : ''}">${row.kronecker}</td>
        <td class="mono ${row.best === 'fourier' ? 'best-val' : ''}">${row.fourier}</td>
      `);
      tbody.appendChild(tr);
    });
  }
  const wtbody = $('#order-worked-tbody');
  if (wtbody && D.orderSensitivityWorked) {
    D.orderSensitivityWorked.forEach(row => {
      const tr = el('tr', '', `
        <td class="mono">${row.pair}</td>
        <td>${row.kind}</td>
        <td class="mono">${row.kronecker.toFixed(3)}</td>
        <td class="mono">${row.fourier.toFixed(3)}</td>
      `);
      wtbody.appendChild(tr);
    });
  }
}

/* ═══════════════════════════════════════════════════════════
   HRR crosstalk table (§18)
   ═══════════════════════════════════════════════════════════ */
function renderCrosstalk() {
  const table = $('#crosstalk-table');
  if (!table || !D.crosstalkTable) return;
  const thead = table.querySelector('thead tr');
  D.crosstalkTable.header.forEach(h => thead.appendChild(el('th', '', h)));
  const tbody = table.querySelector('tbody');
  D.crosstalkTable.rows.forEach(row => {
    const tr = el('tr');
    row.forEach((cell, i) => {
      const td = el('td', i === 0 ? '' : 'mono', cell);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   Updated verdict — theory meets practice (§19)
   ═══════════════════════════════════════════════════════════ */
const STATUS_BADGE = {
  'confirmed':        { cls: 'badge-green',  label: 'CONFIRMED' },
  'confirmed-hard':   { cls: 'badge-yellow', label: 'CONFIRMED (the hard way)' },
  'not-exercised':    { cls: 'badge-red',    label: 'NOT EXERCISED' },
};
function renderUpdatedVerdict() {
  const tbody = $('#updated-verdict-tbody');
  if (!tbody || !D.updatedVerdict) return;
  D.updatedVerdict.forEach(row => {
    const b = STATUS_BADGE[row.status] || { cls: 'badge-blue', label: row.status };
    const tr = el('tr', '', `
      <td>${row.claim} <span class="mono text-muted">${row.ref}</span></td>
      <td><span class="badge ${b.cls}">${b.label}</span></td>
      <td>${row.note}</td>
    `);
    tbody.appendChild(tr);
  });
}

/* ═══════════════════════════════════════════════════════════
   References
   ═══════════════════════════════════════════════════════════ */
function renderReferences() {
  const list = $('#reference-list');
  if (!list || !D.references) return;
  D.references.forEach(ref => {
    const body = ref.href
      ? `${ref.text} <a href="${ref.href}" target="_blank" rel="noopener">↗</a>`
      : ref.text;
    list.appendChild(el('div', 'ref-item', body));
  });
}

/* ═══════════════════════════════════════════════════════════
   Interactive Kernel Explorer
   K(delta) = sum_{i=0}^{m-1} cos(omega_i * delta)
   standard: omega_i = 1 / base^(2i/d_p)
   narrow:   omega_i = omega_low * (1 + i/(m-1)), omega_low = 1/base
   Exact formulas from fourier_embeddings/model/codecs.py
   ═══════════════════════════════════════════════════════════ */
function computeOmegas(dp, schedule, base) {
  const m = Math.floor(dp / 2);
  const omegas = [];
  if (schedule === 'standard') {
    for (let i = 0; i < m; i++) omegas.push(1 / Math.pow(base, (2 * i) / dp));
  } else {
    const omegaLow = 1 / base;
    for (let i = 0; i < m; i++) omegas.push(omegaLow * (1 + i / Math.max(m - 1, 1)));
  }
  return omegas;
}

function kernelValue(omegas, delta) {
  let s = 0;
  for (const w of omegas) s += Math.cos(w * delta);
  return s;
}

function initKernelWidget() {
  const canvas = $('#kernel-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpSlider = $('#kw-dp-slider');
  const dpVal = $('#kw-dp-val');
  const rangeSlider = $('#kw-range-slider');
  const rangeVal = $('#kw-range-val');
  const btnStandard = $('#kw-btn-standard');
  const btnNarrow = $('#kw-btn-narrow');
  const btnBoth = $('#kw-btn-both');

  let mode = 'both'; // 'standard' | 'narrow' | 'both'

  function isDark() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
  }

  function draw() {
    const dp = parseInt(dpSlider.value, 10);
    const range = parseInt(rangeSlider.value, 10);
    dpVal.textContent = `d_p=${dp} (m=${dp / 2} bands)`;
    rangeVal.textContent = `±${range}`;

    const W = canvas.width = canvas.clientWidth * 2;
    const H = canvas.height = 260 * 2;
    ctx.clearRect(0, 0, W, H);

    const dark = isDark();
    const gridColor = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)';
    const axisColor = dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.22)';
    const textColor = dark ? '#94a3b8' : '#64748b';

    const padL = 54, padR = 20, padT = 20, padB = 36;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const m = dp / 2;
    const kMax = m; // K(0) = m
    const kMin = -kMax * 0.5;

    const xForDelta = (delta) => padL + ((delta + range) / (2 * range)) * plotW;
    const yForK = (k) => padT + (1 - (k - kMin) / (kMax - kMin)) * plotH;

    // grid
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let gx = -range; gx <= range; gx += Math.max(1, Math.round(range / 8))) {
      const x = xForDelta(gx);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    }
    ctx.font = '20px JetBrains Mono, monospace';
    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    for (let gx = -range; gx <= range; gx += Math.max(1, Math.round(range / 4))) {
      ctx.fillText(String(gx), xForDelta(gx), padT + plotH + 26);
    }
    ctx.textAlign = 'right';
    ctx.fillText(kMax.toFixed(1), padL - 10, yForK(kMax) + 6);
    ctx.fillText('0', padL - 10, yForK(0) + 6);

    // axes
    ctx.strokeStyle = axisColor;
    ctx.beginPath(); ctx.moveTo(padL, yForK(0)); ctx.lineTo(padL + plotW, yForK(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.stroke();

    // Kronecker delta kernel (spike at 0) — dashed reference
    ctx.strokeStyle = dark ? 'rgba(148,163,184,0.55)' : 'rgba(100,116,139,0.55)';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padL, yForK(0));
    ctx.lineTo(padL + plotW, yForK(0));
    ctx.stroke();
    ctx.setLineDash([]);
    // spike
    ctx.strokeStyle = dark ? '#e2e8f0' : '#334155';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(xForDelta(0), yForK(0));
    ctx.lineTo(xForDelta(0), yForK(kMax));
    ctx.stroke();

    function plotKernel(schedule, color) {
      const omegas = computeOmegas(dp, schedule, D.kernelBase || 10000);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const steps = 400;
      for (let s = 0; s <= steps; s++) {
        const delta = -range + (2 * range * s) / steps;
        const k = kernelValue(omegas, delta);
        const x = xForDelta(delta);
        const y = yForK(k);
        if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (mode === 'standard' || mode === 'both') plotKernel('standard', '#6366f1');
    if (mode === 'narrow' || mode === 'both') plotKernel('narrow', '#f43f5e');
  }

  [dpSlider, rangeSlider].forEach(s => s && s.addEventListener('input', draw));
  function setMode(next) {
    mode = next;
    [btnStandard, btnNarrow, btnBoth].forEach(b => b && b.classList.remove('active'));
    ({ standard: btnStandard, narrow: btnNarrow, both: btnBoth }[next])?.classList.add('active');
    draw();
  }
  btnStandard && btnStandard.addEventListener('click', () => setMode('standard'));
  btnNarrow && btnNarrow.addEventListener('click', () => setMode('narrow'));
  btnBoth && btnBoth.addEventListener('click', () => setMode('both'));

  window.addEventListener('resize', draw);
  const themeBtn = $('#themeToggle');
  themeBtn && themeBtn.addEventListener('click', () => setTimeout(draw, 50));

  draw();
}

/* ═══════════════════════════════════════════════════════════
   NAV — active section on scroll
   ═══════════════════════════════════════════════════════════ */
function initNav() {
  const navItems = $$('.nav-item[data-target]');
  const ids = ['overview', 'kronecker', 'foundations', 'design', 'complementarity', 'setup', 'results', 'probes', 'verdict'];
  const sections = ids.map(id => document.getElementById(id));

  window.addEventListener('scroll', () => {
    let current = 0;
    sections.forEach((sec, i) => {
      if (sec && sec.getBoundingClientRect().top < 120) current = i;
    });
    navItems.forEach((n, i) => n.classList.toggle('active', i === current));
  }, { passive: true });

  navItems.forEach(n => {
    n.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.getElementById(n.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   Hero stat count-up
   ═══════════════════════════════════════════════════════════ */
function animateNum(elemSel, target, decimals, duration = 1100, suffix = '') {
  const elem = $(elemSel);
  if (!elem) return;
  const startTime = performance.now();
  function step(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = target * eased;
    const shown = decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString();
    elem.textContent = shown + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ═══════════════════════════════════════════════════════════
   BOOT
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  renderTheoryVerdict();
  renderParamTheoryTable();
  renderSetup();
  renderParamMeasured();
  renderFinalPerf();
  renderCollisions();
  renderOrderSensitivity();
  renderCrosstalk();
  renderUpdatedVerdict();
  renderReferences();
  initKernelWidget();
  initNav();

  setTimeout(() => {
    animateNum('#stat-codec-cut', 50, 0, 1100, '%');
    animateNum('#stat-fourier-loss', 0.2181, 4);
    animateNum('#stat-kronecker-loss', 0.2197, 4);
    animateNum('#stat-pairs', 500, 0);
    animateNum('#stat-tokens', 327.68, 2, 1100, 'M');
  }, 150);
});
