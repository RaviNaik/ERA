/* ============================================================
   SESSION 10 — The Training Loop — APP
   Renders every widget from window.SESSION_DATA. No build step,
   no external libraries.
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

/* ═══════════════════════ Hero stats ═══════════════════════ */
function renderHeroStats() {
  const wrap = $('#hero-stats');
  if (!wrap) return;
  D.heroStats.forEach(s => {
    wrap.appendChild(el('div', 'hero-stat',
      `<span class="val ${s.cls}">${esc(s.val)}</span><span class="lbl">${esc(s.lbl)}</span>`));
  });
}

/* ═══════════════════════ Six task cards ══════════════════ */
function renderTasks() {
  const grid = $('#task-grid');
  if (!grid) return;
  D.tasks.forEach(item => {
    const card = el('div', 'seven-card task-card');
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

/* ═══════════════════════ Block diagram ═══════════════════ */
function renderBlock() {
  const wrap = $('#block-diagram');
  if (!wrap) return;
  const s = D.setup;
  wrap.innerHTML = `
    <div class="bd-row"><span class="bd-pill ghost">token emb [V, ${s.nEmbd}] + pos emb [${s.blockSize}, ${s.nEmbd}]</span></div>
    <div class="bd-stream">
      <div class="bd-branch">
        <span class="bd-op">LayerNorm</span>
        <span class="bd-op accent">causal MHSA · ${s.nHead} heads · SDPA</span>
        <span class="bd-plus">+ residual</span>
      </div>
      <div class="bd-branch">
        <span class="bd-op">LayerNorm</span>
        <span class="bd-op accent">GELU MLP · 4C = ${fmt(s.mlpHidden)}</span>
        <span class="bd-plus">+ residual</span>
      </div>
      <div class="bd-label">× ${s.nLayer} blocks · pre-norm</div>
    </div>
    <div class="bd-row"><span class="bd-pill">final LayerNorm → hidden [B, T, ${s.nEmbd}]</span></div>
    <div class="bd-row"><span class="bd-pill out">lm_head (tied to wte) → logits [B, T, ${s.vocab}]</span></div>
  `;
}

/* ═══════════════════════ Setup table ════════════════════ */
function renderSetup() {
  const t = $('#setup-table');
  if (!t) return;
  const s = D.setup;
  const rows = [
    ['Hardware', s.hardware],
    ['Dataset', `${s.dataset} · ${fmt(s.trainChars)} train chars · vocab ${s.vocab}`],
    ['Model', `nanoGPT decoder, from scratch — learned pos emb, LayerNorm, GELU MLP, tied head`],
    ['Config', `block ${s.blockSize} · ${s.nLayer} layers · ${s.nHead} heads · n_embd ${s.nEmbd}`],
    ['Parameters', `${fmt(s.paramsTotal)} total · ${fmt(s.paramsNonEmbedding)} non-embedding (the N in 6N)`],
    ['Optimizer', s.optimizer],
    ['Sanity anchor', `ln(${s.vocab}) = ${s.lnVocab} — where an untrained char model lands`],
  ];
  t.innerHTML = rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('');
}

/* ═══════════════════════ Task 1 — shape walk ════════════ */
function renderShapeWalk() {
  const wrap = $('#shape-walk');
  if (!wrap) return;
  wrap.innerHTML = D.shapeWalk.map((st, i) => `
    <div class="spine-node ${st.c}">
      <span class="spine-k">${esc(st.k)}</span>
      <span class="spine-s mono">${esc(st.s)}</span>
    </div>${i < D.shapeWalk.length - 1 ? '<span class="spine-arrow">→</span>' : ''}
  `).join('');

  const dm = D.dims;
  $('#shape-legend').innerHTML =
    `<code>B=${dm.B}</code> batch · <code>T=${dm.T}</code> positions · <code>C=${dm.C}</code> residual width · ` +
    `<code>H=${dm.H}</code> heads · <code>Dh=${dm.Dh}</code> head width · <code>F=${dm.F}</code> MLP hidden · ` +
    `<code>V=${dm.V}</code> vocab. Hover a node: ` +
    D.shapeWalk.map(s => `<br/><code>${esc(s.k)}</code> — ${esc(s.m)}`).join('');

  $('#byhand-badge').textContent = `by-hand = fused to ${D.byHandDiff}`;
}

function renderParamTable() {
  const t = $('#param-table');
  if (!t) return;
  t.innerHTML = `<thead><tr><th>parameter</th><th>shape</th><th>#</th><th>maps</th></tr></thead><tbody>${
    D.params.map(p => `<tr>
      <td class="mono">${esc(p.name)}</td>
      <td class="shape mono">${esc(p.shape)}</td>
      <td class="mono">${fmt(p.n)}</td>
      <td>${esc(p.maps)}</td>
    </tr>`).join('')
  }</tbody>`;
  $('#param-total').textContent = fmt(D.setup.paramsTotal);
}

/* ═══════════════════════ Task 2 — gradient check ════════ */
function renderGradCheck() {
  const g = D.gradCheck;
  if (!g) return;
  $('#gc-param').textContent = g.param;
  $('#gc-badge').textContent = `agree to ${g.decimals} decimals`;
  const cells = [
    ['L(w + h)', g.lPlus.toFixed(12)],
    ['L(w − h)', g.lMinus.toFixed(12)],
    ['analytic  dL/dw  (backward)', g.analytic.toFixed(12)],
    ['numeric  dL/dw  (central diff)', g.numeric.toFixed(12)],
    ['absolute difference', g.absDiff],
    ['relative difference', g.relDiff],
  ];
  $('#gc-panel').innerHTML = cells.map(([k, v]) =>
    `<div class="gc-cell"><div class="gc-k">${esc(k)}</div><div class="gc-v">${esc(v)}</div></div>`).join('') +
    `<div class="gc-cell hi"><div class="gc-k">matching decimals</div><div class="gc-v">${g.decimals}</div></div>`;
  $('#gc-formula').innerHTML =
    `weight value <code>${g.w0}</code>, step <code>h = ${g.h}</code>, fp64 deepcopy, dropout off. ` +
    `<code>dL/dw ≈ (L(w+h) − L(w−h)) / 2h</code> — no <code>backward()</code> involved in the numeric side.`;
}

/* ═══════════════════════ Task 3 — accumulation ═════════ */
function renderAccum() {
  const a = D.accum;
  if (!a) return;
  const max = Math.max(a.correct, a.wrong);
  $('#acc-bars').innerHTML =
    `<div class="acc-row"><span>correct (token-weighted)</span>
       <div class="acc-track"><div class="acc-fill good" style="width:${(a.correct / max * 100).toFixed(1)}%"></div></div>
       <span class="acc-val">${a.correct.toFixed(2)}</span></div>
     <div class="acc-row"><span>wrong (avg of averages)</span>
       <div class="acc-track"><div class="acc-fill bad" style="width:${(a.wrong / max * 100).toFixed(1)}%"></div></div>
       <span class="acc-val">${a.wrong.toFixed(2)}</span></div>`;
  $('#acc-static-badge').textContent = `${a.relErrPct}% off`;
  $('#acc-static-note').innerHTML =
    `token counts <code>(${a.tokenCounts.join(', ')})</code>, mean losses <code>(${a.meanLosses.map(x => x.toFixed(1)).join(', ')})</code>. ` +
    `correct <code>(4·2 + 4·2 + 2·5) / 10 = 2.60</code>; wrong <code>(2 + 2 + 5) / 3 = 3.00</code>.`;

  $('#acc-real-badge').textContent = `+${a.gapPct}% val loss`;
  $('#acc-real-table').innerHTML =
    `<thead><tr><th>after ${a.steps} steps</th><th>correct</th><th>wrong</th></tr></thead><tbody>
      <tr><td>final val loss</td><td class="mono ok">${a.finalValCorrect.toFixed(4)}</td><td class="mono bad">${a.finalValWrong.toFixed(4)}</td></tr>
      <tr><td>gap at the last eval</td><td class="mono dim">—</td><td class="mono bad">+${a.gapPct}%</td></tr>
    </tbody>`;
}

/* ═══════════════════════ Task 4 — grad norm lead ══════ */
function renderGradNorm() {
  const g = D.gradnorm;
  if (!g) return;
  $('#gn-badge').textContent = `leads by ${g.leadSteps} step`;
  $('#lead-strip').innerHTML = `
    <div class="lead-seg"><span class="n">${g.injectStep - 1}</span>normal · grad norm ${g.normBefore}</div>
    <div class="lead-seg spike"><span class="n">${g.injectStep}</span>poisoned batch · grad norm <strong>${g.normAfter}</strong> (${g.normJump}×)</div>
    <div class="lead-seg rise"><span class="n">${g.lossRoseStep}</span>EMA loss rises &gt;5% — one step later</div>
  `;
  $('#gn-note').innerHTML =
    `${g.nSteps}-step run, EMA β = ${g.emaBeta}. The grad norm spikes <strong>on</strong> the poisoned step ` +
    `(${g.spikeStep}); the smoothed loss only reacts at step ${g.lossRoseStep}. Lead = ${g.lossRoseStep} − ${g.spikeStep} = <strong>${g.leadSteps} step</strong>. ` +
    `See the twin-axis plot in the Charts section.`;
}

/* ═══════════════════════ Task 5 — MFU ═════════════════ */
function renderMfu() {
  const m = D.mfu;
  if (!m) return;
  $('#mfu-badge').textContent = `MFU ${m.mfuPct}%`;
  $('#mfu-dist-badge').textContent = `${m.distanceToFortyPts} points to 40%`;

  const pct = m.achievedTflops / m.peakTflops * 100;
  const fill = $('#mfu-fill');
  fill.style.width = pct.toFixed(1) + '%';
  fill.setAttribute('data-label', `${m.achievedTflops} TFLOP/s achieved — MFU ${m.mfuPct}%`);
  const tgt = el('div', 'mfu-target');
  tgt.style.left = '40%';
  $('#mfu-track') ? $('#mfu-track').appendChild(tgt) : fill.parentElement.appendChild(tgt);
  $('#mfu-scale').innerHTML =
    `<span>0</span><span></span><span>${m.peakTflops} peak · ${esc(m.peakSource)}</span>`;

  const fl = [
    ['N (non-embedding)', fmt(m.N)],
    ['6N', fmt(m.sixN)],
    ['attention term', fmt(m.attnTerm)],
    ['flops / token', fmt(m.flopsPerToken)],
    ['dtype', m.dtype],
    ['mean step time', m.stepMs + ' ms'],
    ['tokens / sec', fmt(m.tokensPerSec)],
    ['achieved', m.achievedTflops + ' TFLOP/s'],
  ];
  $('#mfu-flops').innerHTML = fl.map(([k, v]) =>
    `<div class="fl"><div class="fl-k">${esc(k)}</div><div class="fl-v">${esc(v)}</div></div>`).join('');

  $('#mfu-costs').innerHTML = m.costs.map((c, i) =>
    `<div class="comment-card"><span class="comment-n mono">0${i + 1}</span><h4>${esc(c.h)}</h4><p>${esc(c.p)}</p></div>`).join('');
}

/* ═══════════════════════ Task 6 — floats ═════════════ */
function bitRow(bits, cls) {
  return bits.split('').map(b => `<span class="bit ${cls}">${b}</span>`).join('');
}
function renderFloats() {
  const f = D.floats;
  if (!f) return;
  $('#floats-grid').innerHTML = f.formats.map(fm => `
    <div class="float-row">
      <div class="float-head">
        <span class="float-name">${esc(fm.name)} <span>(${esc(fm.layout)})</span></span>
        <span class="float-err text-rose">rel error ${esc(fm.relErr)}</span>
      </div>
      <div class="bits">
        ${bitRow(fm.sign, 's')}
        <span style="width:6px"></span>
        ${bitRow(fm.exp, 'e')}
        <span style="width:6px"></span>
        ${bitRow(fm.mant, 'm')}
      </div>
      <div class="float-val">= ${fm.value.toFixed(12)}   ·   exponent field ${parseInt(fm.exp, 2)} = −4 + bias   ·   mantissa ${fm.mant.length} bits</div>
    </div>
  `).join('');

  if ($('#float-nan-note')) $('#float-nan-note').textContent = f.nanNote || '';
  $('#float-extras').innerHTML =
    `<thead><tr><th>input</th><th>fp8 E4M3 bits</th><th>value</th><th>note</th></tr></thead><tbody>${
      f.extras.map(x => `<tr>
        <td class="mono">${esc(x.in)}</td>
        <td class="mono tok">${esc(x.bits)}</td>
        <td class="mono">${esc(x.value)}</td>
        <td class="dim">${esc(x.note)}</td>
      </tr>`).join('')
    }</tbody>`;

  $('#float-answer').textContent = f.answer;
}

/* ═══════════════════════ Sanity + figures ════════════ */
function renderSanity() {
  const s = D.sanity;
  if (!s) return;
  $('#sanity-badge').textContent = `${s.startLoss} → ${s.endLoss} in ${s.steps} steps`;
  $('#sanity-table').innerHTML =
    `<thead><tr><th>${s.steps} steps · batch ${s.batch} × seq ${s.seq} · clip ${s.clip} from step one</th><th>value</th></tr></thead><tbody>
      <tr><td>start loss</td><td class="mono">${s.startLoss} <span class="dim">(ln V = ${s.lnVocab})</span></td></tr>
      <tr><td>end loss</td><td class="mono ok">${s.endLoss}</td></tr>
      <tr><td>wall time</td><td class="mono">${s.wallSec} s</td></tr>
    </tbody>`;
  $('#sanity-sample').textContent = s.sample;
}
function renderFigures() {
  const grid = $('#figure-grid');
  if (!grid) return;
  D.figures.forEach(f => {
    grid.appendChild(el('figure', 'figure-card',
      `<img src="${f.src}" alt="${esc(f.cap)}" loading="lazy" /><figcaption>${esc(f.cap)}</figcaption>`));
  });
}

/* ═══════════════════════ Boot ═══════════════════════ */
function boot() {
  renderHeroStats();
  renderTasks();
  renderBlock();
  renderSetup();
  renderShapeWalk();
  renderParamTable();
  renderGradCheck();
  renderAccum();
  renderGradNorm();
  renderMfu();
  renderFloats();
  renderSanity();
  renderFigures();
}
document.addEventListener('DOMContentLoaded', boot);
