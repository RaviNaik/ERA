/* ============================================================
   SESSION 12 — 32 Virtual GPUs, ZeRO-0..3 — APP
   Renders window.SESSION_DATA + three live widgets: the ZeRO
   stage explorer, the memory/communication calculator, and the
   parameter-ownership visualizer. No build step, no libraries.
   ============================================================ */

const D = window.SESSION_DATA || {};
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const GIB = 1024 ** 3;

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function highlightJson(json) {
  return esc(json)
    .replace(/"([a-zA-Z0-9_]+)":/g, '<span class="k">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
    .replace(/: (true|false|[0-9.e+-]+)/g, ': <span class="n">$1</span>');
}

/* ═══════════════════════ Hero ═══════════════════════ */
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

/* ═══════════════════════ 1. Data parallelism ═══════════════════════ */
function renderDp() {
  const flow = $('#dp-flow');
  if (flow) {
    flow.innerHTML = D.dp.steps.map(s => `
      <div class="flow-step">
        <div class="flow-n">${s.n}</div>
        <div class="flow-body"><h4>${esc(s.t)}</h4><p>${esc(s.d)}</p></div>
      </div>`).join('');
  }
  const e = D.dp.equivalence;
  $('#dp-equivalence').innerHTML = [
    ['World size', e.worldSize], ['Samples / rank', e.samplesPerRank],
    ['Reference (global-batch) loss', e.referenceLoss.toFixed(9)],
    ['Mean per-rank loss', e.meanRankLoss.toFixed(9)],
    ['Max gradient difference', e.maxGradDiff.toExponential(3)],
    ['Max post-step parameter difference', e.maxParamDiff.toExponential(3)],
  ].map(([k, v]) => `<tr><th>${k}</th><td class="mono">${v}</td></tr>`).join('');
  $('#dp-findings').innerHTML = D.dp.findings.map(f => `<li>${esc(f)}</li>`).join('');
}

/* ═══════════════════════ 2. Landscape ═══════════════════════ */
function renderLandscape() {
  $('#landscape-note').textContent = D.landscapeNote;
  $('#landscape-grid').innerHTML = D.landscape.map(l => `
    <div class="landscape-card reveal-item ${l.name.startsWith('ZeRO') ? 'zero' : ''}">
      <h4>${esc(l.name)}</h4>
      <span class="landscape-tag">${esc(l.tag)}</span>
      <div class="landscape-row"><b>Shards</b>${esc(l.shards)}</div>
      <div class="landscape-row good"><b style="color:var(--green)">Good for</b>${esc(l.good)}</div>
      <div class="landscape-row bad"><b style="color:var(--rose)">Watch out for</b>${esc(l.bad)}</div>
    </div>`).join('');
}

/* ═══════════════════════ 3. ZeRO stage explorer ═══════════════════════ */
let activeStage = 0;
function renderStagePills() {
  $('#stage-pills').innerHTML = D.zeroStages.map(s => `
    <button class="stage-pill ${s.id === activeStage ? 'active' : ''}" data-stage="${s.id}">
      <div class="sp-name">${esc(s.name)}</div>
      <div class="sp-sub">${esc(s.subtitle)}</div>
    </button>`).join('');
  $$('.stage-pill').forEach(btn => btn.addEventListener('click', () => {
    activeStage = Number(btn.dataset.stage);
    renderStagePills();
    renderStageDetail();
  }));
}
function renderStageDetail() {
  const s = D.zeroStages.find(x => x.id === activeStage);
  if (!s) return;
  $('#stage-title').textContent = `${s.name} — ${s.subtitle}`;
  $('#stage-chips').innerHTML =
    s.replicated.map(x => `<span class="state-chip replicated">replicated: ${esc(x)}</span>`).join('') +
    s.sharded.map(x => `<span class="state-chip sharded">sharded: ${esc(x)}</span>`).join('');
  $('#stage-explainer').textContent = s.whatHappens;
  $('#stage-formula').textContent = s.formula;
  $('#stage-collectives').innerHTML = s.collectives.map(c =>
    `<li class="${c.includes('NEW') ? 'new' : ''}">${esc(c)}</li>`).join('');
  $('#stage-comm-badge').textContent = `${s.commRatio.toFixed(1)}x baseline comms`;
  $('#stage-config').innerHTML = highlightJson(s.configJson);
  $('#stage-config-note').textContent = s.configNote;
  $('#stage-proscons').innerHTML = `
    <div class="pc-col pros"><h5>Advantages</h5><ul>${s.pros.map(p => `<li>${esc(p)}</li>`).join('')}</ul></div>
    <div class="pc-col cons"><h5>Shortcomings</h5><ul>${s.cons.map(c => `<li>${esc(c)}</li>`).join('')}</ul></div>
  `;
}

/* ═══════════════════════ 4. Memory & comm calculator ═══════════════════════ */
const calcState = { modelB: 7, worldSize: 8, gpuGiB: 24, precision: 'mixed', activationGiB: 4, layerPct: 2 };

function bytesForStage(stageId, P, N, cfg, activationBytes, layerFrac) {
  const paramTotal = P * cfg.param, gradTotal = P * cfg.grad, optTotal = P * (cfg.master + cfg.moment);
  const param = stageId >= 3 ? paramTotal / N : paramTotal;
  const grad = stageId >= 2 ? gradTotal / N : gradTotal;
  const opt = stageId >= 1 ? optTotal / N : optTotal;
  const transient = stageId === 3 ? (P * layerFrac) * cfg.param * (N - 1) / N : 0;
  return { param, grad, opt, transient, persistent: param + grad + opt, peak: param + grad + opt + transient + activationBytes };
}

function renderCalcControls() {
  const wrap = $('#calc-controls');
  wrap.innerHTML = `
    <div class="calc-field">
      <label>Model size <span class="calc-val" id="v-modelB">${calcState.modelB.toFixed(1)}B params</span></label>
      <input type="range" id="c-modelB" min="0.1" max="200" step="0.1" value="${calcState.modelB}" />
    </div>
    <div class="calc-field">
      <label>World size (GPUs) <span class="calc-val" id="v-worldSize">${calcState.worldSize}</span></label>
      <input type="range" id="c-worldSize" min="1" max="512" step="1" value="${calcState.worldSize}" />
    </div>
    <div class="calc-field">
      <label>GPU memory <span class="calc-val" id="v-gpuGiB">${calcState.gpuGiB} GiB</span></label>
      <input type="range" id="c-gpuGiB" min="8" max="141" step="1" value="${calcState.gpuGiB}" />
    </div>
    <div class="calc-field">
      <label>Activation memory / GPU <span class="calc-val" id="v-activationGiB">${calcState.activationGiB} GiB</span></label>
      <input type="range" id="c-activationGiB" min="0" max="40" step="0.5" value="${calcState.activationGiB}" />
    </div>
    <div class="calc-field">
      <label>Largest layer (% of params, ZeRO-3 gather) <span class="calc-val" id="v-layerPct">${calcState.layerPct}%</span></label>
      <input type="range" id="c-layerPct" min="0.1" max="20" step="0.1" value="${calcState.layerPct}" />
    </div>
    <div class="calc-field">
      <label>Precision</label>
      <select id="c-precision">
        <option value="mixed">${D.formulaBytes.mixedPrecision.label}</option>
        <option value="fullFp32">${D.formulaBytes.fullFp32.label}</option>
      </select>
    </div>
  `;
  $('#c-precision').value = calcState.precision === 'mixed' ? 'mixed' : 'fullFp32';
  const bind = (id, key, fmt) => $(id).addEventListener('input', (e) => {
    calcState[key] = e.target.type === 'range' ? Number(e.target.value) : e.target.value;
    $(`#v-${key}`) && ($(`#v-${key}`).textContent = fmt ? fmt(calcState[key]) : calcState[key]);
    renderCalcResults();
  });
  bind('#c-modelB', 'modelB', v => v.toFixed(1) + 'B params');
  bind('#c-worldSize', 'worldSize', v => v);
  bind('#c-gpuGiB', 'gpuGiB', v => v + ' GiB');
  bind('#c-activationGiB', 'activationGiB', v => v + ' GiB');
  bind('#c-layerPct', 'layerPct', v => v + '%');
  $('#c-precision').addEventListener('change', (e) => { calcState.precision = e.target.value; renderCalcResults(); });
}

function renderCalcResults() {
  const P = calcState.modelB * 1e9;
  const N = calcState.worldSize;
  const gpuBytes = calcState.gpuGiB * GIB;
  const activationBytes = calcState.activationGiB * GIB;
  const layerFrac = calcState.layerPct / 100;
  const cfg = D.formulaBytes[calcState.precision === 'mixed' ? 'mixedPrecision' : 'fullFp32'];

  const rows = D.zeroStages.map(s => {
    const b = bytesForStage(s.id, P, N, cfg, activationBytes, layerFrac);
    const fits = b.peak <= gpuBytes * 0.92;
    return { ...s, bytes: b, fits };
  });

  const maxPeak = Math.max(...rows.map(r => r.bytes.peak), gpuBytes);
  const colors = { 0: '--rose', 1: '--amber', 2: '--sky', 3: '--green' };
  const recommended = rows.find(r => r.fits);

  $('#calc-results').innerHTML = rows.map(r => `
    <div class="calc-row ${r.fits ? 'fits' : 'overflow'} ${recommended && r.id === recommended.id ? 'recommended' : ''}">
      <div class="cr-stage">${esc(r.name)}</div>
      <div class="cr-bar-track"><div class="cr-bar-fill" style="width:${Math.min(100, r.bytes.peak / maxPeak * 100).toFixed(1)}%;background:${cssVar(colors[r.id])}"></div></div>
      <div class="cr-mem">${(r.bytes.peak / GIB).toFixed(1)} GiB</div>
      <div class="cr-verdict" style="color:${r.fits ? cssVar('--green') : cssVar('--rose')}">${r.fits ? 'fits' : 'too big'}</div>
    </div>`).join('');

  $('#calc-recommend').innerHTML = recommended
    ? `<b>Recommended: ${esc(recommended.name)}.</b> It is the lowest-numbered (least communication) stage whose estimated peak
       (${(recommended.bytes.peak / GIB).toFixed(1)} GiB) fits within ${calcState.gpuGiB} GiB per GPU, leaving ~8% headroom
       for allocator overhead. Communication for this stage is ${recommended.commRatio.toFixed(1)}x the ZeRO-0 baseline.`
    : `<b>No stage fits.</b> Even ZeRO-3 needs ${(rows[3].bytes.peak / GIB).toFixed(1)} GiB/GPU at N=${N}.
       Increase world size, add CPU/NVMe offload (ZeRO-Infinity), or combine with tensor/pipeline parallelism.`;
}

/* ═══════════════════════ 5. Ownership visualizer ═══════════════════════ */
function renderOwnership() {
  const pSlider = $('#own-p'), nSlider = $('#own-n');
  function draw() {
    const P = Number(pSlider.value), N = Number(nSlider.value);
    $('#own-p-val').textContent = P;
    $('#own-n-val').textContent = N;
    const pad = (N - (P % N)) % N;
    const padded = P + pad;
    const shard = padded / N;
    const strip = $('#ownership-strip');
    strip.innerHTML = '';
    for (let rank = 0; rank < N; rank++) {
      const hue = Math.round((rank / N) * 300);
      const cell = el('div', 'ownership-cell');
      const width = shard / padded * 100;
      cell.style.width = width + '%';
      cell.style.background = `hsl(${hue}, 65%, ${document.documentElement.getAttribute('data-theme') === 'light' ? 55 : 50}%)`;
      cell.title = `rank ${rank}: elements [${Math.min(rank * shard, P)}, ${Math.min((rank + 1) * shard, P)})`;
      strip.appendChild(cell);
    }
    $('#ownership-legend').textContent =
      `${P} parameters padded to ${padded} (+${pad}) so they split evenly across ${N} ranks → `
      + `${shard} elements/rank, exactly balanced (max−min shard size = 0). Hover a segment for its index range.`;
  }
  pSlider.addEventListener('input', draw);
  nSlider.addEventListener('input', draw);
  draw();
}

/* ═══════════════════════ 6. Real experiment ═══════════════════════ */
function renderExperiment() {
  const ex = D.experiment;
  $('#exp-setup').innerHTML = ex.setup.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('');

  $('#exp-correctness').innerHTML = `<thead><tr><th>Stage</th><th>max |distributed − reference|</th></tr></thead><tbody>${
    ex.correctness.map(c => `<tr><td>${esc(c.stage)}</td><td class="mono tok ok">${esc(c.diff)}</td></tr>`).join('')
  }</tbody>`;

  $('#exp-memory-table').innerHTML = `<thead><tr><th>Stage</th><th>Params</th><th>Gradients</th><th>Optimizer</th><th>Transient gather</th><th>Persistent total</th></tr></thead><tbody>${
    ex.memoryTable.map(r => `<tr>
      <td>${esc(r.label)}</td><td class="mono">${r.param} B</td><td class="mono">${r.grad} B</td>
      <td class="mono">${r.opt} B</td><td class="mono ${r.transient ? 'tok bad' : 'dim'}">${r.transient || '—'} B</td>
      <td class="mono tok ok">${r.total} B</td>
    </tr>`).join('')
  }</tbody>`;

  const maxTotal = Math.max(...ex.memoryTable.map(r => r.param + r.grad + r.opt + r.transient));
  $('#exp-memory-bars').innerHTML = ex.memoryTable.map(r => {
    const segs = [
      { v: r.param, color: cssVar('--indigo'), label: 'param' },
      { v: r.grad, color: cssVar('--amber'), label: 'grad' },
      { v: r.opt, color: cssVar('--rose'), label: 'optimizer' },
      { v: r.transient, color: cssVar('--sky'), label: 'transient' },
    ];
    return `<div style="margin-bottom:10px">
      <div class="mem-label"><span>${esc(r.label)}</span><span class="dim mono">${r.param + r.grad + r.opt + r.transient} bytes/rank</span></div>
      <div class="stack-bar">${segs.map(s => `<div class="stack-seg" style="width:${(s.v / maxTotal * 100).toFixed(2)}%;background:${s.color}"></div>`).join('')}</div>
    </div>`;
  }).join('') + `<div class="stack-legend">
    <span class="lg"><i style="background:${cssVar('--indigo')}"></i>parameters</span>
    <span class="lg"><i style="background:${cssVar('--amber')}"></i>gradients</span>
    <span class="lg"><i style="background:${cssVar('--rose')}"></i>optimizer (m, v)</span>
    <span class="lg"><i style="background:${cssVar('--sky')}"></i>transient gather (ZeRO-3 only)</span>
  </div>`;

  $('#exp-reduction-badge').textContent = ex.reduction.split(' — ')[0];

  $('#exp-formula-check').innerHTML = `<thead><tr><th>Stage</th><th>Measured</th><th>Formula-predicted</th><th>Match</th></tr></thead><tbody>${
    ex.formulaCheck.map(r => `<tr><td>${esc(r.stage)}</td><td class="mono">${r.measured} B</td><td class="mono">${r.predicted} B</td><td class="mono tok ok">${r.measured === r.predicted ? '✓ exact' : '✗'}</td></tr>`).join('')
  }</tbody>`;

  const maxRatio = Math.max(...ex.commRatios.map(r => r.ratio));
  $('#exp-comm-bars').innerHTML = ex.commRatios.map(r => `
    <div class="mem-row">
      <div class="mem-label"><span>${esc(r.stage)}</span><span class="dim mono">${r.ratio.toFixed(1)}x</span></div>
      <div class="mem-track"><div class="mem-fill ${r.ratio > 1 ? 'rose' : 'green'}" style="width:${(r.ratio / maxRatio * 100).toFixed(1)}%"></div></div>
    </div>`).join('');

  $('#exp-figures').innerHTML = `<figure class="figure-card reveal-item">
    <img src="../parallelization_experiments/assets/measured_memory_by_stage.png" alt="Real measured per-rank memory across all four ZeRO stages" loading="lazy" />
    <figcaption>Generated by the real 32-process notebook run — see ../parallelization_experiments/assets/</figcaption>
  </figure>`;
}

/* ═══════════════════════ 7. Decision guide ═══════════════════════ */
function renderGuide() {
  $('#guide-checklist').innerHTML = D.decisionChecklist.map((c, i) => `
    <div class="checklist-item"><div class="checklist-n">${i + 1}</div><p>${esc(c)}</p></div>`).join('');

  $('#scenario-grid').innerHTML = D.scenarios.map(s => `
    <div class="scenario-card reveal-item">
      <h4>${esc(s.title)}</h4>
      <div class="scenario-numbers">${esc(s.numbers)}</div>
      <span class="badge badge-blue scenario-verdict">${esc(s.verdict)}</span>
      <p class="scenario-reason">${esc(s.reasoning)}</p>
    </div>`).join('');
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
      if (e.isIntersecting) { e.target.classList.add('in-view'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
  items.forEach(i => obs.observe(i));
}

/* ═══════════════════════ Boot ═══════════════════════ */
function boot() {
  renderHeroStats();
  renderDp();
  renderLandscape();
  renderStagePills();
  renderStageDetail();
  renderCalcControls();
  renderCalcResults();
  renderOwnership();
  renderExperiment();
  renderGuide();
  initReveal();
}
document.addEventListener('DOMContentLoaded', boot);
