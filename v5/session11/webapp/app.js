/* ============================================================
   SESSION 11 — Optimizers by Hand — APP
   Renders every widget from window.SESSION_DATA. No build step,
   no external libraries. Charts are hand-drawn inline SVG.
   ============================================================ */

const D = window.SESSION_DATA || {};
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmt = (n) => Number(n).toLocaleString();
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ═══════════════════════ Hero stats ═══════════════════════ */
function renderHeroStats() {
  const wrap = $('#hero-stats');
  if (!wrap) return;
  D.heroStats.forEach((s, i) => {
    const card = el('div', 'hero-stat reveal-item');
    card.style.transitionDelay = (i * 70) + 'ms';
    card.innerHTML = `<span class="val ${s.cls}">${s.val}</span><span class="lbl">${esc(s.lbl)}</span>`;
    wrap.appendChild(card);
  });
}

/* ═══════════════════════ Five task cards ══════════════════ */
function renderTasks() {
  const grid = $('#task-grid');
  if (!grid) return;
  D.tasks.forEach(item => {
    const card = el('div', 'seven-card task-card reveal-item');
    card.innerHTML = `
      <div class="seven-top">
        <span class="seven-n mono">${item.n}</span>
        <span class="seven-verdict badge badge-blue">${esc(item.verdict)}</span>
      </div>
      <h4 class="seven-title">${esc(item.title)}</h4>
      <p class="seven-headline mono">${esc(item.headline)}</p>
      <p class="seven-detail">${esc(item.detail)}</p>
    `;
    card.addEventListener('click', () => card.classList.toggle('open'));
    grid.appendChild(card);
  });
}

/* ═══════════════════════ Setup table + param bars ═══════ */
function renderSetup() {
  const t = $('#setup-table');
  if (t) {
    const s = D.setup;
    const rows = [
      ['Hardware', s.hardware],
      ['Dataset', `${s.dataset} · ${fmt(s.totalChars)} chars · vocab ${s.vocab} · ${s.trainSplit} split`],
      ['Model', `nanoGPT decoder — block ${s.blockSize} · ${s.nLayer} layers · ${s.nHead} heads · n_embd ${s.nEmbd}`],
      ['Parameters', `${fmt(s.paramsTotal)} total (10.77M — matches the assignment exactly)`],
      ['Optimizer', s.optimizer],
      ['Tracking', `${s.tracking} (${s.aimRuns} runs)`],
    ];
    t.innerHTML = rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`).join('');
  }
  const pw = $('#param-bars');
  if (pw) {
    pw.innerHTML = D.paramTable.map(p => `
      <div class="mem-row">
        <div class="mem-label">${esc(p.name)} <span class="dim mono">${fmt(p.n)} · ${p.pct}</span></div>
        <div class="mem-track"><div class="mem-fill indigo" data-w="${p.pct}" style="width:0%"></div></div>
      </div>`).join('');
    requestAnimationFrame(() => animateFills(pw));
  }
}

function animateFills(scope) {
  $$('.mem-fill[data-w], .acc-fill[data-w], .ratio-fill[data-w]', scope).forEach((f, i) => {
    setTimeout(() => { f.style.width = f.dataset.w; }, 60 + i * 60);
  });
}

/* ═══════════════════════ Task 1 — Adam by hand ═══════════ */
function renderAdam() {
  const a = D.adam;
  if (!a) return;
  $('#adam-setup').innerHTML =
    `w₀ = <code>${a.w0}</code> · gradients <code>[${a.grads.join(', ')}]</code> · ` +
    `lr <code>${a.lr}</code> · β₁ <code>${a.beta1}</code> · β₂ <code>${a.beta2}</code> · ε <code>${a.eps}</code>`;

  $('#adam-trace').innerHTML = `<thead><tr><th>t</th><th>m</th><th>v</th><th>m̂</th><th>v̂</th><th>step Δₜ</th><th>w after</th></tr></thead><tbody>${
    a.trace.map(r => `<tr>
      <td class="mono">${r.t}</td><td class="mono">${r.m.toFixed(6)}</td><td class="mono">${r.v}</td>
      <td class="mono">${r.mhat.toFixed(6)}</td><td class="mono">${r.vhat.toFixed(6)}</td>
      <td class="mono">${r.step}</td><td class="mono tok ok">${r.w.toFixed(8)}</td>
    </tr>`).join('')
  }</tbody>`;

  $('#adam-agree').innerHTML = a.agreement.map(g => `
    <div class="gc-cell"><div class="gc-k">${esc(g.variant)}</div><div class="gc-v ok">${esc(g.diff)}</div></div>
  `).join('');

  $('#adam-decay').innerHTML = `<thead><tr><th>t</th><th>Adam (no decay)</th><th>Adam + L2</th><th>AdamW (decoupled)</th></tr></thead><tbody>${
    a.decayCompare.map(r => `<tr><td class="mono">${r.t}</td><td class="mono">${r.adam.toFixed(8)}</td><td class="mono">${r.l2.toFixed(8)}</td><td class="mono tok ok">${r.adamw.toFixed(8)}</td></tr>`).join('')
  }</tbody>`;

  $('#adam-findings').innerHTML = a.findings.map(f => `<li>${f.replace(/(m̂₁ = g₁|√v̂₁ = \|g₁\|)/g, '<code>$1</code>')}</li>`).join('');
}

/* ═══════════════════════ Task 2 — bias correction ════════ */
function renderBias() {
  const b = D.bias;
  if (!b) return;
  $('#bias-horizons').innerHTML = `<thead><tr><th>β₂</th><th>within 5%</th><th>within 1%</th><th>context</th></tr></thead><tbody>${
    b.horizons.map(h => `<tr><td class="mono">${h.beta2}</td><td class="mono">${h.within5}</td><td class="mono tok">${h.within1}</td><td class="dim">${esc(h.ctx)}</td></tr>`).join('')
  }</tbody>`;

  $('#bias-real-title').textContent = b.realWeight.name + `  (β₂ = ${b.realWeight.beta2})`;
  $('#bias-real').innerHTML = `<thead><tr><th>t</th><th>c(t) predicted</th><th>|step ratio| measured</th><th>|w_BC − w_noBC|</th></tr></thead><tbody>${
    b.realWeight.rows.map(r => `<tr><td class="mono">${r.t}</td><td class="mono">${r.c.toFixed(3)}</td><td class="mono">${r.ratio.toFixed(3)}</td><td class="mono tok bad">${r.gap}</td></tr>`).join('')
  }</tbody>`;
  $('#bias-real-note').innerHTML = `After 20 steps the two weights differ by <strong>${b.realWeight.gapPctOfMovement}%</strong> of how far this weight has moved at all.`;

  const max = Math.max(...b.step1Multiplier.map(m => m.mult));
  $('#bias-mult').innerHTML = b.step1Multiplier.map(m => `
    <div class="acc-row">
      <span>β₂ = ${m.beta2}</span>
      <div class="acc-track"><div class="acc-fill ${m.mult >= 1 ? 'good' : 'bad'}" data-w="${(m.mult / max * 100).toFixed(1)}%" style="width:0%"></div></div>
      <span class="acc-val">${m.mult.toFixed(2)}× <span class="dim">(${m.note})</span></span>
    </div>`).join('');
  requestAnimationFrame(() => animateFills($('#bias-mult').parentElement));

  $('#bias-findings').innerHTML = b.findings.map(f => `<li>${esc(f)}</li>`).join('');
}

/* ═══════════════════════ Task 3 — update/weight ratio ═════ */
function renderRatio() {
  const r = D.ratio;
  if (!r) return;
  const maxR = 0.012;
  $('#ratio-table').innerHTML = `<thead><tr><th>layer</th><th>ρ@5</th><th>ρ@50 (peak)</th><th>ρ@150</th><th>ρ@299</th><th></th></tr></thead><tbody>${
    r.layers.map(l => `<tr class="${l.cold ? 'cold' : ''}">
      <td>${esc(l.layer)}</td><td class="mono">${l.r5}</td><td class="mono">${l.r50}</td><td class="mono">${l.r150}</td>
      <td class="mono tok ${l.cold ? 'bad' : 'ok'}">${l.r299}</td>
      <td style="width:90px"><div class="mem-track" style="height:8px"><div class="mem-fill ${l.cold ? 'rose' : 'green'}" data-w="${Math.min(100, parseFloat(l.r299) / maxR * 100)}%" style="width:0%"></div></div></td>
    </tr>`).join('')
  }</tbody>`;
  requestAnimationFrame(() => animateFills($('#ratio-table')));
  $('#ratio-global').textContent = r.globalFinal;

  $('#warmup-table').innerHTML = `<thead><tr><th>warmup</th><th>argmax ρ (step)</th><th>ρ peak</th><th>final val loss</th></tr></thead><tbody>${
    r.warmupRuns.map(w => `<tr><td class="mono">${w.warmup}</td><td class="mono tok ok">${w.argmax}</td><td class="mono">${w.peak}</td><td class="mono">${w.finalVal.toFixed(4)}</td></tr>`).join('')
  }</tbody>`;

  $('#ratio-findings').innerHTML = r.findings.map(f => `<li>${esc(f)}</li>`).join('');
}

/* ═══════════════════════ Inline SVG line chart ═══════════ */
function lineChart(container, opts) {
  const W = container.clientWidth || 680;
  const H = opts.height || 300;
  const pad = { t: 16, r: 16, b: 30, l: 52 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const xs = opts.x;
  const allY = opts.series.flatMap(s => s.y);
  let ymin = opts.ymin != null ? opts.ymin : Math.min(...allY);
  let ymax = opts.ymax != null ? opts.ymax : Math.max(...allY);
  const padY = (ymax - ymin) * 0.1 || 0.01;
  ymin -= padY; ymax += padY;
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const logX = !!opts.logX;
  const xval = v => logX ? Math.log10(v) : v;
  const xminL = xval(xmin), xmaxL = xval(xmax);
  const X = v => pad.l + (xval(v) - xminL) / (xmaxL - xminL) * iw;
  const Y = v => pad.t + (1 - (v - ymin) / (ymax - ymin)) * ih;

  const grid = cssVar('--border') || '#333';
  const axis = cssVar('--text-muted') || '#888';

  const yticks = 4;
  let g = '';
  for (let i = 0; i <= yticks; i++) {
    const v = ymin + (ymax - ymin) * i / yticks;
    const y = Y(v);
    g += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="${grid}" stroke-width="1"/>`;
    g += `<text x="${pad.l - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="${axis}">${v.toFixed(opts.yfmt != null ? opts.yfmt : 2)}</text>`;
  }
  xs.forEach((v, i) => {
    if (opts.xEvery && i % opts.xEvery !== 0) return;
    const x = X(v);
    g += `<text x="${x}" y="${H - 8}" text-anchor="middle" font-size="9.5" fill="${axis}">${opts.xfmt ? opts.xfmt(v) : v}</text>`;
  });

  let paths = '';
  opts.series.forEach(s => {
    const pts = s.y.map((v, i) => [X(xs[i]), Y(v)]);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    paths += `<path class="chart-line" d="${d}" fill="none" stroke="${cssVar(s.color)}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} opacity="${s.opacity != null ? s.opacity : 1}"/>`;
    if (s.dots) {
      pts.forEach(p => { paths += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3" fill="${cssVar(s.color)}"/>`; });
    }
    if (s.star != null) {
      const sx = X(s.star.x), sy = Y(s.star.y);
      paths += `<path transform="translate(${sx},${sy})" d="M0,-7L2,-2L7,-2L3,1L5,7L0,3L-5,7L-3,1L-7,-2L-2,-2Z" fill="${cssVar(s.color)}" stroke="#000" stroke-width="0.5" opacity="0.95"/>`;
    }
  });

  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" style="display:block">${g}${paths}</svg>`;
}

/* ═══════════════════════ Task 4 — cosine vs WSD ═══════════ */
function renderSchedules() {
  const s = D.schedules;
  if (!s) return;
  $('#sched-meta').innerHTML = `warmup <code>${s.warmup}</code> · base_lr <code>${s.baseLr}</code> · total_steps <code>${s.totalSteps}</code>`;

  $('#sched-table').innerHTML = `<thead><tr><th>checkpoint</th><th>val loss</th><th>lr there</th></tr></thead><tbody>${
    s.checkpoints.map(c => `<tr class="${c.tag === 'best' ? 'best' : ''}">
      <td>${esc(c.name)}${c.tag === 'best' ? ' <span class="badge badge-green" style="margin-left:6px">best</span>' : ''}</td>
      <td class="mono">${c.loss.toFixed(3)}</td><td class="mono dim">${esc(c.lr)}</td>
    </tr>`).join('')
  }</tbody>`;

  const chart = $('#sched-chart');
  if (chart) {
    lineChart(chart, {
      x: s.lrCurve.steps, ylabel: 'learning rate', yfmt: 4, xEvery: 2,
      series: [
        { y: s.lrCurve.cosine, color: '--indigo', width: 2.4 },
        { y: s.lrCurve.wsd, color: '--amber', width: 2.4 },
      ],
    });
  }

  $('#sched-findings').innerHTML = s.findings.map(f => `<li>${esc(f)}</li>`).join('');
}

/* ═══════════════════════ Task 5 — LR sweep vs width ═══════ */
function renderLrSweep() {
  const l = D.lrsweep;
  if (!l) return;

  $('#lrsweep-table').innerHTML = `<thead><tr><th>width</th><th>params</th><th>fitted η*</th><th>grid-best η*</th><th>val loss</th><th>2-seed spread</th></tr></thead><tbody>${
    l.widths.map(w => `<tr><td class="mono">${w.width}</td><td class="mono">${w.params}</td><td class="mono tok ok">${w.fitted}</td><td class="mono dim">${w.gridBest}</td><td class="mono">${w.loss}</td><td class="mono ${w.spread.includes('0.03') ? 'bad' : ''}">${w.spread}</td></tr>`).join('')
  }</tbody>`;

  const chart = $('#lrsweep-chart');
  if (chart) {
    const colors = { 256: '--sky', 512: '--purple', 1024: '--rose' };
    lineChart(chart, {
      x: l.coarseGrid, logX: true, ylabel: 'val loss', yfmt: 2, xfmt: v => v.toExponential(0),
      series: Object.keys(l.coarseLoss).map(w => ({
        y: l.coarseLoss[w], color: colors[w], width: 2.2, dots: true,
      })),
    });
  }

  const pw = D.lrsweep.powerLaw;
  $('#lrsweep-power').innerHTML =
    `<div class="fl"><div class="fl-k">power-law exponent b</div><div class="fl-v">${pw.b}</div></div>` +
    `<div class="fl"><div class="fl-k">fit residual σ</div><div class="fl-v">${pw.sigma}</div></div>` +
    `<div class="fl"><div class="fl-k">η*(4096), power-law fit</div><div class="fl-v tok ok">${pw.pred4096}</div></div>` +
    `<div class="fl"><div class="fl-k">±2σ band</div><div class="fl-v dim">${pw.band[0]} – ${pw.band[1]}</div></div>` +
    `<div class="fl"><div class="fl-k">endpoint-only bracket</div><div class="fl-v dim">${pw.endpointOnly}</div></div>`;

  const rec = D.lrsweep.recommendation;
  $('#lrsweep-rec').innerHTML =
    `<div class="rec-value">${rec.value}</div>` +
    `<div class="rec-label">recommended starting LR at width 4096</div>` +
    `<div class="rec-chips">${rec.confirmSweep.map(v => `<span class="chip mono">${v}</span>`).join('')}</div>` +
    `<div class="rec-conf badge badge-red">confidence: ${rec.confidence}</div>`;

  $('#lrsweep-findings').innerHTML = l.findings.map(f => `<li>${esc(f)}</li>`).join('');
}

/* ═══════════════════════ Figures + commentary ═══════════ */
function renderFigures() {
  const grid = $('#figure-grid');
  if (!grid) return;
  D.figures.forEach(f => {
    grid.appendChild(el('figure', 'figure-card reveal-item',
      `<img src="${f.src}" alt="${esc(f.cap)}" loading="lazy" /><figcaption>${esc(f.cap)}</figcaption>`));
  });
}
function renderCommentary() {
  const grid = $('#commentary-grid');
  if (!grid) return;
  D.commentary.forEach((c, i) => {
    grid.appendChild(el('div', 'comment-card reveal-item',
      `<span class="comment-n mono">0${i + 1}</span><h4>${esc(c.h)}</h4><p>${esc(c.p)}</p>`));
  });
}

/* ═══════════════════════ Scroll-reveal ═══════════════════ */
function initReveal() {
  const items = $$('.reveal-item, .card');
  if (!('IntersectionObserver' in window) || !items.length) {
    items.forEach(i => i.classList.add('in-view'));
    return;
  }
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('in-view');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
  items.forEach(i => obs.observe(i));
}

/* ═══════════════════════ Boot ═══════════════════════ */
function boot() {
  renderHeroStats();
  renderTasks();
  renderSetup();
  renderAdam();
  renderBias();
  renderRatio();
  renderSchedules();
  renderLrSweep();
  renderFigures();
  renderCommentary();
  initReveal();
}
document.addEventListener('DOMContentLoaded', boot);
window.addEventListener('resize', () => {
  clearTimeout(window.__s11rz);
  window.__s11rz = setTimeout(() => { renderSchedules(); renderLrSweep(); }, 200);
});
