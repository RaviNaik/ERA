/* ============================================================
   SESSION 9 — The Loss Harness — APP
   Renders every widget from window.SESSION_DATA. No build step,
   no external libraries. Charts are hand-drawn inline SVG so
   they stay crisp and theme-aware.
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
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* ═══════════════════════ Hero stats ═══════════════════════ */
function renderHeroStats() {
  const wrap = $('#hero-stats');
  if (!wrap) return;
  D.heroStats.forEach(s => {
    wrap.appendChild(el('div', 'hero-stat',
      `<span class="val ${s.cls}">${esc(s.val)}</span><span class="lbl">${esc(s.lbl)}</span>`));
  });
}

/* ═══════════════════════ Shift tables ═════════════════════ */
function renderShiftTables() {
  const good = $('#shift-table');
  const bug = $('#shift-bug-table');
  if (good) {
    good.innerHTML = `<thead><tr><th>pos</th><th>input</th><th>→ target</th></tr></thead><tbody>${
      D.shiftRows.map(r => `<tr><td class="mono dim">${r.pos}</td><td class="mono tok">${esc(JSON.stringify(r.in))}</td><td class="mono tok ok">${esc(JSON.stringify(r.tgt))}</td></tr>`).join('')
    }</tbody>`;
  }
  if (bug) {
    bug.innerHTML = `<thead><tr><th>pos</th><th>input</th><th>→ wrong target</th></tr></thead><tbody>${
      D.shiftRows.slice(0, 6).map(r => `<tr><td class="mono dim">${r.pos}</td><td class="mono tok">${esc(JSON.stringify(r.in))}</td><td class="mono tok bad">${esc(JSON.stringify(r.in))}</td></tr>`).join('')
    }</tbody>`;
  }
}

/* ═══════════════════════ Loss spine ══════════════════════ */
function renderSpine() {
  const wrap = $('#spine-diagram');
  if (!wrap) return;
  const stages = [
    { k: 'tokens', s: '[B, T]', c: 'sky' },
    { k: 'trunk', s: '8 blocks', c: 'indigo' },
    { k: 'hidden h', s: '[B, T, 512]', c: 'indigo' },
    { k: 'output head', s: 'W · [V, D]', c: 'purple' },
    { k: 'logits z', s: '[B, T, 50257]', c: 'purple' },
    { k: 'softmax', s: 'gaps → ratios', c: 'amber' },
    { k: 'probs p', s: '[B, T, V]', c: 'amber' },
    { k: 'cross-entropy', s: '−log p(next)', c: 'rose' },
    { k: 'loss L', s: 'one scalar', c: 'green' },
  ];
  wrap.innerHTML = stages.map((st, i) => `
    <div class="spine-node ${st.c}">
      <span class="spine-k">${esc(st.k)}</span>
      <span class="spine-s mono">${esc(st.s)}</span>
    </div>${i < stages.length - 1 ? '<span class="spine-arrow">→</span>' : ''}
  `).join('');
}

/* ═══════════════════════ Block diagram ═══════════════════ */
function renderBlock() {
  const wrap = $('#block-diagram');
  if (!wrap) return;
  wrap.innerHTML = `
    <div class="bd-row"><span class="bd-pill ghost">token emb [V,D] + pos emb</span></div>
    <div class="bd-stream">
      <div class="bd-branch">
        <span class="bd-op">RMSNorm</span>
        <span class="bd-op accent">causal MHSA · 8 heads</span>
        <span class="bd-plus">+ residual</span>
      </div>
      <div class="bd-branch">
        <span class="bd-op">RMSNorm</span>
        <span class="bd-op accent">SwiGLU FFN · d_ff 1408</span>
        <span class="bd-plus">+ residual</span>
      </div>
      <div class="bd-label">× 8 blocks · pre-norm</div>
    </div>
    <div class="bd-row"><span class="bd-pill">final RMSNorm → hidden h [B,T,512]</span></div>
    <div class="bd-row"><span class="bd-pill out">output head → logits [B,T,50257]</span></div>
  `;
}

/* ═══════════════════════ Setup table ═════════════════════ */
function renderSetup() {
  const t = $('#setup-table');
  if (!t) return;
  const s = D.setup;
  const rows = [
    ['Hardware', s.hardware],
    ['Dataset', `${s.dataset} · ${s.docs.toLocaleString()} docs · ${s.tokens.toLocaleString()} tokens`],
    ['Tokenizer', `${s.tokenizer} · V = ${s.vocab.toLocaleString()}`],
    ['Trunk', `d_model ${s.dModel} · ${s.nLayers} layers · ${s.nHeads} heads · d_ff ${s.dFF}`],
    ['Trunk params', `${s.trunkParams.toLocaleString()} (tok_emb ${s.tokEmbParams.toLocaleString()})`],
    ['Norm / layout', s.norm],
    ['FFN', s.ffn],
    ['Training', `seq ${s.seqLen} · batch ${s.batch} (${s.tokensPerStep.toLocaleString()} tok/step) · ${s.steps.toLocaleString()} steps`],
    ['Optimizer', s.optimizer],
    ['Wall clock', `baseline ${Math.round(s.baselineWallSec / 60)} min · MTP ${Math.round(s.mtpWallSec / 60)} min`],
  ];
  t.innerHTML = rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('');
}

/* ═══════════════════════ Seven numbers ══════════════════ */
function renderSeven() {
  const grid = $('#seven-grid');
  if (!grid) return;
  D.sevenNumbers.forEach(item => {
    const card = el('div', 'seven-card');
    card.innerHTML = `
      <div class="seven-top">
        <span class="seven-n mono">${item.n}</span>
        <span class="seven-verdict badge badge-${item.tag === 'perplexity' || item.tag === 'shift' ? 'green' : 'blue'}">${esc(item.verdict)}</span>
      </div>
      <h4 class="seven-title">${esc(item.title)}</h4>
      <p class="seven-headline mono">${esc(item.headline)}</p>
      <p class="seven-detail">${esc(item.detail)}</p>
    `;
    card.addEventListener('click', () => card.classList.toggle('open'));
    grid.appendChild(card);
  });
}

/* ═══════════════════════ Boundary: untrained vs trained ══ */
function renderBoundaryDetail() {
  const wrap = $('#boundary-detail');
  if (!wrap || !D.boundaryTrained) return;
  const b = D.boundaryTrained;
  const row = (label, u, t, cls = '') => `<tr><td>${esc(label)}</td><td class="mono">${u}</td><td class="mono ${cls}">${t}</td></tr>`;
  wrap.innerHTML = `<table class="mini-table"><thead><tr><th>on the same two documents</th><th>untrained (§3.4)</th><th>trained t+1 head (§6.2)</th></tr></thead><tbody>
    ${row('mean loss, all positions', b.untrainedMean.toFixed(4), b.trainedMean.toFixed(4))}
    ${row('mean loss, boundary position masked', b.untrainedMasked.toFixed(4), b.trainedMasked.toFixed(4))}
    ${row('the boundary position itself', b.untrainedBoundary.toFixed(4), `<strong>${b.trainedBoundary.toFixed(4)}</strong>`, 'bad')}
    ${row('Δ from masking one position', '−0.0038', `<strong>${b.trainedDelta.toFixed(4)}</strong>`)}
    ${row('boundary loss ÷ mean-of-the-rest', b.untrainedRatio.toFixed(2) + '×', `<strong>${b.trainedRatio.toFixed(2)}×</strong>`)}
  </tbody></table>`;
}

/* ═══════════════════════ Memory bars ════════════════════ */
function renderMemBars() {
  const wrap = $('#mem-bars');
  if (!wrap) return;
  const m = D.memory;
  const max = m.ordinaryGiB;
  const bar = (label, gib, cls) => `
    <div class="mem-row">
      <span class="mem-label">${label}</span>
      <div class="mem-track"><div class="mem-fill ${cls}" style="width:${(gib / max * 100).toFixed(1)}%"></div></div>
      <span class="mem-val mono">${gib.toFixed(2)} GiB</span>
    </div>`;
  const brk = (path) => (m.breakdown || []).filter(b => b.path === path)
    .map(b => `<tr><td class="mono">${esc(b.item)}</td><td class="mono">${b.gib.toFixed(2)}</td><td class="dim">${esc(b.note)}</td></tr>`).join('');
  wrap.innerHTML =
    bar(`ordinary — full [${m.nTokens.toLocaleString()}, 50257] logits + fp32 log-softmax + grad`, m.ordinaryGiB, 'rose') +
    bar(`chunked — one [${m.chunk.toLocaleString()}, 50257] slice, ${m.nPasses} passes`, m.chunkedGiB, 'green') +
    `<div class="mem-ratio">measured ratio <strong>${m.ratio}×</strong> (predicted ${m.predictedOrdinaryGiB} / ${m.predictedChunkedGiB} GiB) · loss agrees to <span class="mono">${m.lossDiff}</span> · gradients to <span class="mono">${m.gradDiff}</span></div>` +
    `<div class="mem-breakdown">
       <div class="mem-bk-title">where the bytes actually go — the naive [N,V] figure is ${m.theoreticalFull}, not the cost</div>
       <table class="mini-table"><thead><tr><th>ordinary path</th><th>GiB</th><th></th></tr></thead><tbody>${brk('ordinary')}</tbody></table>
       <table class="mini-table"><thead><tr><th>chunked path</th><th>GiB</th><th></th></tr></thead><tbody>${brk('chunked')}</tbody></table>
     </div>`;
}

/* ═══════════════════════ MTP head cards ═════════════════ */
function renderMtpHeads() {
  const wrap = $('#mtp-heads');
  if (!wrap) return;
  const m = D.mtp;
  const card = (name, h, settled, cls) => `
    <div class="head-card ${cls}">
      <div class="head-name">${name} <span class="mono dim">predicts ${h.predicts}</span></div>
      <div class="head-loss"><span class="mono">${h.lossStart.toFixed(2)}</span> → <span class="mono dim">${h.lossEnd.toFixed(2)}</span> → <strong class="mono">${settled.toFixed(2)}</strong> <span class="dim">settled</span></div>
      <div class="head-ppl">perplexity ${Math.round(h.pplStart).toLocaleString()} → <strong class="mono">~${Math.round(Math.exp(settled))}</strong></div>
    </div>`;
  wrap.innerHTML =
    card('Head 1', m.head1, m.head1Settled, 'h1') +
    card('Head 2', m.head2, m.head2Settled, 'h2') +
    `<div class="head-card sum">
       <div class="head-name">Sum <span class="mono dim">L1 + L2 · optimised</span></div>
       <div class="head-loss"><span class="mono">${m.sum.lossStart.toFixed(2)}</span> → <span class="mono dim">${m.sum.lossEnd.toFixed(2)}</span> → <strong class="mono">${m.sumSettled.toFixed(2)}</strong></div>
       <div class="head-ppl">standalone baseline (settled) → <strong class="mono">${m.baselineSettled.toFixed(2)}</strong> · gap to MTP head1 <strong class="mono">+${m.head1MinusBaseSettled.toFixed(3)}</strong></div>
     </div>`;
}

/* ═══════════════════════ SVG line chart ═════════════════ */
function lineChart(container, opts) {
  const W = container.clientWidth || 720;
  const H = opts.height || 340;
  const pad = { t: 18, r: 16, b: 34, l: 46 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const xs = opts.x;
  const allY = opts.series.flatMap(s => s.y);
  let ymin = opts.ymin != null ? opts.ymin : Math.min(...allY);
  let ymax = opts.ymax != null ? opts.ymax : Math.max(...allY);
  const padY = (ymax - ymin) * 0.08;
  ymin -= padY; ymax += padY;
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const X = v => pad.l + (v - xmin) / (xmax - xmin) * iw;
  const Y = v => pad.t + (1 - (v - ymin) / (ymax - ymin)) * ih;

  const grid = cssVar('--border');
  const axis = cssVar('--text-muted');

  const yticks = 5, xticks = 5;
  let g = '';
  for (let i = 0; i <= yticks; i++) {
    const v = ymin + (ymax - ymin) * i / yticks;
    const y = Y(v);
    g += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="${grid}" stroke-width="1"/>`;
    g += `<text x="${pad.l - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="${axis}">${v.toFixed(opts.yfmt || 1)}</text>`;
  }
  for (let i = 0; i <= xticks; i++) {
    const v = xmin + (xmax - xmin) * i / xticks;
    const x = X(v);
    g += `<text x="${x}" y="${H - 10}" text-anchor="middle" font-size="10" fill="${axis}">${Math.round(v)}</text>`;
  }

  let paths = '';
  opts.series.forEach(s => {
    const d = s.y.map((v, i) => `${i ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
    paths += `<path d="${d}" fill="none" stroke="${cssVar(s.color)}" stroke-width="${s.width || 2}" stroke-linejoin="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} opacity="${s.opacity || 1}"/>`;
  });

  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" style="display:block">
    ${opts.ylabel ? `<text transform="rotate(-90)" x="${-(pad.t + ih / 2)}" y="12" text-anchor="middle" font-size="10" fill="${axis}">${opts.ylabel}</text>` : ''}
    ${g}${paths}
    <text x="${pad.l + iw / 2}" y="${H - 2}" text-anchor="middle" font-size="10" fill="${axis}">training step</text>
  </svg>`;
}

let curveMode = 'both';
function drawCurve() {
  const c = $('#curve-chart');
  if (!c) return;
  const legend = $('#curve-legend');
  const title = $('#chart-title');
  if (curveMode === 'both') {
    lineChart(c, {
      x: D.steps, ylabel: 'loss (nats)',
      series: [
        { y: D.curveH1, color: '--indigo', width: 2 },
        { y: D.curveH2, color: '--amber', width: 2 },
      ],
    });
    title.textContent = 'Head losses over 4,000 steps';
    legend.innerHTML = `<span class="lg"><i style="background:${cssVar('--indigo')}"></i>Head 1 · t+1 → settled ${D.mtp.head1Settled}</span>
      <span class="lg"><i style="background:${cssVar('--amber')}"></i>Head 2 · t+2 → settled ${D.mtp.head2Settled}</span>`;
  } else if (curveMode === 'gap') {
    const gap = D.curveH2.map((v, i) => +(v - D.curveH1[i]).toFixed(3));
    lineChart(c, {
      x: D.steps, ylabel: 'L2 − L1 (nats)', ymin: -0.2, yfmt: 2,
      series: [{ y: gap, color: '--rose', width: 2.5 }, { y: D.steps.map(() => 0), color: '--text-muted', width: 1, dash: '4 4' }],
    });
    title.textContent = 'The gap: L2 − L1 over training';
    legend.innerHTML = `<span class="lg"><i style="background:${cssVar('--rose')}"></i>gap opens 0.06 → settled +${D.mtp.gapSettled} (final +${D.mtp.gapEnd.toFixed(2)}); never negative across 4,000 steps</span>`;
  } else {
    lineChart(c, {
      x: D.steps, ylabel: 'loss (nats)',
      series: [
        { y: D.curveSum, color: '--purple', width: 2 },
        { y: D.curveH1, color: '--indigo', width: 1.5, opacity: 0.55 },
        { y: D.curveBase, color: '--green', width: 1.5, dash: '5 4' },
      ],
    });
    title.textContent = 'Optimised sum vs. the standalone baseline';
    legend.innerHTML = `<span class="lg"><i style="background:${cssVar('--purple')}"></i>Sum L1+L2 → settled ${D.mtp.sumSettled}</span>
      <span class="lg"><i style="background:${cssVar('--indigo')}"></i>MTP head 1 (untied) → ${D.mtp.head1Settled}</span>
      <span class="lg"><i style="background:${cssVar('--green')}"></i>baseline (tied, solo) → ${D.mtp.baselineSettled}</span>`;
  }
}

function wireCurveTabs() {
  $$('#curve-tabs .tab-btn').forEach(b => {
    b.addEventListener('click', () => {
      $$('#curve-tabs .tab-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      curveMode = b.dataset.mode;
      drawCurve();
    });
  });
}

/* ═══════════════════════ Commentary ═════════════════════ */
function renderCommentary() {
  const grid = $('#commentary-grid');
  if (!grid) return;
  D.commentary.forEach((c, i) => {
    grid.appendChild(el('div', 'comment-card',
      `<span class="comment-n mono">0${i + 1}</span><h4>${esc(c.h)}</h4><p>${esc(c.p)}</p>`));
  });
}

/* ═══════════════════════ Figures ═══════════════════════ */
function renderFigures() {
  const grid = $('#figure-grid');
  if (!grid) return;
  D.figures.forEach(f => {
    grid.appendChild(el('figure', 'figure-card',
      `<img src="${f.src}" alt="${esc(f.cap)}" loading="lazy" /><figcaption>${esc(f.cap)}</figcaption>`));
  });
}

/* ═══════════════════════ Boot ═════════════════════════ */
function boot() {
  renderHeroStats();
  renderShiftTables();
  renderSpine();
  renderBlock();
  renderSetup();
  renderSeven();
  renderBoundaryDetail();
  renderMemBars();
  renderMtpHeads();
  renderCommentary();
  renderFigures();
  wireCurveTabs();
  drawCurve();
}
document.addEventListener('DOMContentLoaded', boot);

let _rt;
window.addEventListener('resize', () => { clearTimeout(_rt); _rt = setTimeout(drawCurve, 150); });
window.__redrawCharts = drawCurve;
