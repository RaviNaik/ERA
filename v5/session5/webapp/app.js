// ============================================================
// app.js — V5 Mixture & Curriculum Dashboard
// Session 5 Assignment · ERA V5
// ============================================================

'use strict';

// ── State ───────────────────────────────────────────────────
let state = {
  mixMode:         'pretrain',   // 'pretrain' | 'anneal'
  selectedLane:    null,
  selectedStage:   0,
};

// ── DOM Helpers ─────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls)  e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

// ── Status helpers ───────────────────────────────────────────
function statusBadge(s) {
  const map = {
    covered:     ['badge-green',  '✅ Covered'],
    repeat:      ['badge-yellow', '⚠ Repeat'],
    'repeat+synth': ['badge-yellow', '⚠ Repeat+Synth'],
    'must-synth':['badge-red',   '🔴 Must Synthesize'],
  };
  const [cls, txt] = map[s] || ['badge-blue', s];
  return `<span class="badge ${cls}">${txt}</span>`;
}

// ── Number formatting ────────────────────────────────────────
function fmtB(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'T';
  if (n >= 1)    return n.toFixed(0) + 'B';
  return (n * 1000).toFixed(0) + 'M';
}

// ═══════════════════════════════════════════════════════════
// SECTION 1 — Donut Chart + Lane Table
// ═══════════════════════════════════════════════════════════

function drawDonut(canvasId, data, size = 280) {
  const canvas = $('' + '#' + canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = size / 2, cy = size / 2;
  const outer = size * 0.44;
  const inner = outer * 0.62;

  ctx.clearRect(0, 0, size, size);

  const total = data.reduce((s, d) => s + d.share, 0);
  let angle = -Math.PI / 2;
  const gap = 0.015;

  data.forEach((d) => {
    const slice = (d.share / total) * (Math.PI * 2) - gap;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outer, angle + gap / 2, angle + gap / 2 + slice);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();

    // Inner ring highlight
    ctx.beginPath();
    ctx.arc(cx, cy, inner, angle + gap / 2, angle + gap / 2 + slice);
    ctx.arc(cx, cy, inner - 6, angle + gap / 2 + slice, angle + gap / 2, true);
    ctx.closePath();
    ctx.fillStyle = d.color + '30';
    ctx.fill();

    angle += slice + gap;
  });

  // Center hole
  ctx.beginPath();
  ctx.arc(cx, cy, inner - 6, 0, Math.PI * 2);
  ctx.fillStyle = '#050811';
  ctx.fill();
}

function buildLaneTable() {
  const tbody = $('#laneTbody');
  if (!tbody) return;
  const rows = [];
  const data = state.mixMode === 'pretrain' ? DATA.pretrain : DATA.anneal;

  tbody.innerHTML = '';
  data.forEach((lane, i) => {
    const tr = el('tr', 'lane-row');
    tr.dataset.idx = i;

    // For anneal, no supply data
    const isAnneal = state.mixMode === 'anneal';
    const supplyCell = isAnneal
      ? `<td class="text-muted mono" style="font-size:10px;">—</td>
         <td></td>`
      : (() => {
          const pct = Math.min(100, (lane.supply / Math.max(lane.supply, lane.demand2T)) * 100);
          return `<td class="mono" style="font-size:11px;">${fmtB(lane.supply)}</td>
          <td>${statusBadge(lane.status)}</td>`;
        })();

    const demandCell = isAnneal
      ? `<td class="mono text-muted" style="font-size:11px;">${(40 * lane.share / 100).toFixed(1)}B</td>`
      : `<td class="mono" style="font-size:11px;">${fmtB(lane.demand2T)}</td>`;

    tr.innerHTML = `
      <td>
        <div class="lane-name-cell">
          <span class="lane-dot" style="background:${lane.color};"></span>
          <span style="font-size:12px;font-weight:600;">${lane.lane}</span>
        </div>
      </td>
      <td style="font-size:18px;font-weight:800;color:${lane.color};">${lane.share}%</td>
      ${demandCell}
      ${supplyCell}
    `;

    tr.addEventListener('click', () => {
      if (state.selectedLane === i) {
        state.selectedLane = null;
      } else {
        state.selectedLane = i;
      }
      renderLaneDetail(data);
      $$('.lane-row').forEach(r => r.classList.remove('selected'));
      if (state.selectedLane !== null) tr.classList.add('selected');
    });

    tbody.appendChild(tr);
  });
}

function renderLaneDetail(data) {
  const panel = $('#laneDetailPanel');
  if (state.selectedLane === null || !panel) { if (panel) panel.innerHTML = ''; return; }

  const lane = data[state.selectedLane];
  const isAnneal = state.mixMode === 'anneal';

  let supplyHtml = '';
  if (!isAnneal) {
    const pct = Math.min(100, (lane.supply / Math.max(lane.supply, lane.demand2T)) * 100);
    const gapB = Math.max(0, lane.demand2T - lane.supply);
    supplyHtml = `
      <div style="margin-top:12px;">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Supply vs Demand</div>
        <div class="supply-bar-wrap" style="margin-bottom:4px;">
          <div style="width:70px;font-size:11px;color:var(--text-secondary);">Real supply</div>
          <div class="supply-bar-track" style="flex:1;">
            <div class="supply-bar-fill" style="width:${pct}%;background:${lane.color};"></div>
          </div>
          <div class="mono" style="font-size:11px;width:60px;text-align:right;color:var(--text-secondary);">${fmtB(lane.supply)}</div>
        </div>
        <div class="supply-bar-wrap">
          <div style="width:70px;font-size:11px;color:var(--text-secondary);">Demand 2T</div>
          <div class="supply-bar-track" style="flex:1;">
            <div class="supply-bar-fill" style="width:100%;background:${lane.color}60;"></div>
          </div>
          <div class="mono" style="font-size:11px;width:60px;text-align:right;color:var(--text-secondary);">${fmtB(lane.demand2T)}</div>
        </div>
        ${gapB > 0 ? `<div style="margin-top:8px;font-size:11px;color:var(--rose);">Gap: ${fmtB(gapB)} must be synthesized or repeated</div>` : ''}
      </div>
    `;
  }

  const datasetsHtml = lane.datasets
    ? lane.datasets.map(d => `<span class="chip">${d}</span>`).join('')
    : '';
  const benchmarksHtml = lane.benchmarks
    ? lane.benchmarks.map(b => `<span class="badge badge-indigo" style="background:rgba(99,102,241,0.15);color:#a5b4fc;border:1px solid rgba(99,102,241,0.25);">${b}</span>`).join('')
    : '';

  panel.innerHTML = `
    <div class="lane-detail">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span class="lane-dot" style="background:${lane.color};"></span>
            <span style="font-size:15px;font-weight:700;">${lane.lane}</span>
          </div>
          <div style="font-size:24px;font-weight:800;color:${lane.color};letter-spacing:-1px;margin-bottom:4px;">${lane.share}%</div>
        </div>
        ${!isAnneal ? statusBadge(lane.status) : ''}
      </div>
      ${supplyHtml}
      ${benchmarksHtml ? `
        <div style="margin-top:12px;">
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Benchmark Targets</div>
          <div class="detail-chips">${benchmarksHtml}</div>
        </div>
      ` : ''}
      ${datasetsHtml ? `
        <div style="margin-top:12px;">
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Datasets</div>
          <div class="detail-chips">${datasetsHtml}</div>
        </div>
      ` : ''}
    </div>
  `;
}

function renderMixture() {
  const data = state.mixMode === 'pretrain' ? DATA.pretrain : DATA.anneal;
  drawDonut('donutCanvas', data, 280);
  buildLaneTable();
  renderLaneDetail(data);

  const cv = $('#centerVal');
  const cl = $('#centerLbl');
  if (cv) cv.textContent = state.mixMode === 'pretrain' ? '2T' : '40B';
  if (cl) cl.textContent = state.mixMode === 'pretrain' ? 'Total Budget' : 'Anneal Reserve';
}

// ═══════════════════════════════════════════════════════════
// SECTION 2 — Indic Tier Cards
// ═══════════════════════════════════════════════════════════

function renderIndicTiers() {
  const container = $('#indicTierCards');
  if (!container) return;

  container.innerHTML = DATA.indicTiers.map(t => `
    <div class="tier-card ${t.warning ? 'warning-card' : ''}">
      <div class="tier-label" style="color:${t.color};">${t.tier}</div>
      <div class="tier-share-big" style="color:${t.color};">${t.share}%</div>
      <div class="tier-meta">${t.note}</div>
      ${t.warning ? `<div class="warning-banner">⚠️ 14.4× synthesis gap — flagged</div>` : ''}
      <div class="tier-supply-row">
        <span>Real supply: <span class="have">${fmtB(t.realSupplyB)}</span></span>
        <span>Demand: <span class="need">${fmtB(t.demandB)}</span></span>
      </div>
      <div style="margin-top:10px;height:4px;border-radius:2px;background:rgba(255,255,255,0.05);">
        <div style="height:4px;border-radius:2px;background:${t.color};width:${Math.min(100,(t.realSupplyB/t.demandB)*100).toFixed(1)}%;transition:width 0.8s;"></div>
      </div>
    </div>
  `).join('');

  // Bar chart
  const barsContainer = $('#indicBars');
  if (!barsContainer) return;
  const maxVal = Math.max(...DATA.indicTiers.map(t => t.demandB));

  barsContainer.innerHTML = DATA.indicTiers.map(t => `
    <div class="ibar-row">
      <div class="ibar-label">${t.tier.split('—')[0].trim()}</div>
      <div style="flex:1;display:flex;flex-direction:column;gap:4px;">
        <div class="ibar-track">
          <div class="ibar-fill" style="width:${(t.demandB/maxVal*100).toFixed(1)}%;background:${t.color};">Demand ${fmtB(t.demandB)}</div>
        </div>
        <div class="ibar-track">
          <div class="ibar-fill" style="width:${(t.realSupplyB/maxVal*100).toFixed(1)}%;background:${t.color}60;">Supply ${fmtB(t.realSupplyB)}</div>
        </div>
      </div>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════
// SECTION 3 — Protected Floors
// ═══════════════════════════════════════════════════════════

function renderFloors() {
  const grid = $('#floorsGrid');
  if (!grid) return;

  const classes = ['indic-floor', 'agentic-floor'];
  grid.innerHTML = DATA.protectedFloors.map((f, i) => `
    <div class="floor-card ${classes[i]}">
      <div class="floor-name">${f.lane} Lane Floor</div>
      <div class="floor-value">≥ ${f.floor}%</div>
      <div class="floor-reason">${f.reason}</div>
      <div class="floor-mechanism">${f.mechanism}</div>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════
// SECTION 4 — Anneal Reserve
// ═══════════════════════════════════════════════════════════

function renderAnneal() {
  drawDonut('annealDonut', DATA.anneal, 220);

  const reservedDatasets = [
    { name: 'SWE-Gym',           tokens: '150M', lane: 'Agentic', note: '2.4K samples; highest token/sample density' },
    { name: 'SWE-smith (top 5K)', tokens: '~24M', lane: 'Agentic', note: 'Top 5K of 26K trajectories' },
    { name: 'OpenHands rollouts', tokens: '90M',  lane: 'Agentic', note: '10K multi-step agent trajectories' },
    { name: 'OpenThoughts2',      tokens: '3B',   lane: 'Reasoning', note: 'Full dataset reserved for anneal' },
    { name: 'Sangraha verified ≥ 4.0', tokens: '~16B', lane: 'Indic', note: 'Top-quartile educational score shards' },
    { name: 'AON best shards',    tokens: '~10B', lane: 'Reasoning', note: 'Highest quality reasoning traces' },
    { name: 'proof-pile-2',       tokens: '55B',  lane: 'STEM', note: 'Full EleutherAI math corpus' },
  ];

  const laneColors = {
    'Agentic':   '#f43f5e',
    'Reasoning': '#8b5cf6',
    'Indic':     '#f59e0b',
    'STEM':      '#10b981',
  };

  const container = $('#annealDatasets');
  if (!container) return;
  container.innerHTML = reservedDatasets.map(d => `
    <div style="display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--bg-card-alt);border-radius:var(--radius-sm);border:1px solid var(--border);">
      <span class="lane-dot" style="background:${laneColors[d.lane] || '#6366f1'};flex-shrink:0;"></span>
      <div style="flex:1;">
        <div style="font-size:12px;font-weight:600;">${d.name}</div>
        <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">${d.note}</div>
      </div>
      <span class="mono" style="font-size:11px;color:var(--amber);flex-shrink:0;">${d.tokens}</span>
      <span class="badge badge-yellow" style="flex-shrink:0;">${d.lane}</span>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════
// SECTION 5 — Agentic Inventory
// ═══════════════════════════════════════════════════════════

function renderAgentic() {
  const container = $('#agenticItems');
  if (!container) return;
  const maxTokens = Math.max(...DATA.agenticDatasets.map(d => d.tokens));

  container.innerHTML = DATA.agenticDatasets.map(d => {
    const isAnneal = d.anneal === true;
    const isPartial = d.anneal === 'partial';
    const color = isAnneal ? 'var(--amber)' : isPartial ? '#f97316' : 'var(--rose)';
    const barColor = isAnneal ? 'var(--amber)' : 'var(--rose)';
    const tag = isAnneal
      ? '<span class="agentic-anneal-tag">ANNEAL ONLY</span>'
      : isPartial
        ? '<span class="agentic-anneal-tag" style="color:#f97316;border-color:#f97316;background:rgba(249,115,22,0.1);">PARTIAL</span>'
        : '';

    return `
      <div class="agentic-item ${isAnneal ? 'anneal-item' : ''}">
        <div class="agentic-name" style="color:${color};">${d.name}</div>
        <div class="agentic-bar-wrap">
          <div class="agentic-bar-track">
            <div class="agentic-bar-fill" style="width:${(d.tokens/maxTokens*100).toFixed(1)}%;background:${barColor};"></div>
          </div>
          <div class="agentic-token-val">${d.tokens}M</div>
        </div>
        <div class="agentic-tier-badge">
          <span class="badge ${d.tier === 'A' ? 'badge-green' : d.tier === 'D' ? 'badge-purple' : 'badge-blue'}">${d.tier}</span>
        </div>
        ${tag}
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// SECTION 6 — Difficulty Bands
// ═══════════════════════════════════════════════════════════

function renderBands() {
  const grid = $('#bandsGrid');
  if (!grid) return;

  grid.innerHTML = DATA.difficultyBands.map((b, i) => `
    <div class="band-card" style="--band-color:${b.color};"
         onclick="toggleBandDetail(this, ${i})">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${b.color};border-radius:var(--radius-md) var(--radius-md) 0 0;"></div>
      <div class="band-badge" style="color:${b.color};">${b.band}</div>
      <div class="band-level">${b.level}</div>
      <div class="band-tokens mono">~${b.reasoningTokens} reasoning tokens</div>
      <div class="band-example-preview">
        <div class="q">${b.example.problem}</div>
        <div class="a">→ ${b.example.answer}</div>
      </div>
      <div class="band-stage text-muted">${b.stageUse}</div>
    </div>
  `).join('');
}

function toggleBandDetail(card, idx) {
  const existing = $('#bandDetailExpanded');
  if (existing) existing.remove();

  const b = DATA.difficultyBands[idx];
  const detailEl = el('div', '');
  detailEl.id = 'bandDetailExpanded';
  detailEl.style.cssText = 'margin-top:16px;animation:fadeSlideIn 0.2s ease;';
  detailEl.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid ${b.color}40;border-radius:var(--radius-md);padding:24px;">
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:16px;">
        <span style="font-size:28px;font-weight:800;color:${b.color};">${b.band}</span>
        <div>
          <div style="font-size:14px;font-weight:700;">${b.level}</div>
          <div class="mono" style="font-size:11px;color:var(--text-muted);">${b.reasoningTokens} reasoning tokens</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div>
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Problem (masked in loss)</div>
          <div style="background:var(--bg-card-alt);border-radius:6px;padding:12px;font-size:12px;color:var(--text-secondary);line-height:1.6;border-left:3px solid rgba(255,255,255,0.1);">${b.example.problem}</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Reasoning (in loss 🟢)</div>
          <div style="background:var(--bg-card-alt);border-radius:6px;padding:12px;font-size:12px;color:var(--green);line-height:1.6;border-left:3px solid var(--green);">${b.example.reasoning}</div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Answer (in loss 🟢)</div>
        <div style="background:rgba(16,185,129,0.08);border-radius:6px;padding:10px;font-size:13px;font-weight:700;color:var(--green);border:1px solid rgba(16,185,129,0.2);">${b.example.answer}</div>
      </div>
      <div style="margin-top:14px;font-size:11px;color:var(--text-muted);padding:8px;background:var(--bg-card-alt);border-radius:6px;">
        📅 ${b.stageUse}
      </div>
    </div>
  `;
  const grid = $('#bandsGrid');
  grid.parentNode.insertBefore(detailEl, grid.nextSibling);
}

// ═══════════════════════════════════════════════════════════
// SECTION 7 — Curriculum Stages
// ═══════════════════════════════════════════════════════════

const STAGE_COLORS = {
  web: '#6366f1', code: '#22d3ee', indic: '#f59e0b',
  stem: '#10b981', reasoning: '#8b5cf6', longctx: '#06b6d4', agentic: '#f43f5e',
};
const STAGE_WIDTHS = [5, 35, 30, 12, 8]; // visual proportions

function renderCurriculum() {
  const segments = $('#stageSegments');
  const labels   = $('#stageLabels');
  const grid     = $('#stageDetailsGrid');
  if (!segments || !labels || !grid) return;

  const stageColors = ['#6366f1','#22d3ee','#f59e0b','#10b981','#8b5cf6'];

  segments.innerHTML = DATA.curriculumStages.map((s, i) => `
    <div class="stage-seg" style="flex:${STAGE_WIDTHS[i]};background:${stageColors[i]};opacity:0.7;"></div>
  `).join('');

  labels.innerHTML = DATA.curriculumStages.map((s, i) => `
    <div class="stage-lbl" style="color:${stageColors[i]};flex:${STAGE_WIDTHS[i]};font-weight:600;">${s.name}</div>
  `).join('');

  grid.innerHTML = DATA.curriculumStages.map((s, i) => {
    const keys = ['web','code','indic','stem','reasoning','longctx','agentic'];
    const miniBars = keys.map(k => `
      <div class="smbar-row">
        <div class="smbar-label">${k}</div>
        <div class="smbar-track">
          <div class="smbar-fill" style="width:${s[k] || 0}%;background:${STAGE_COLORS[k]};"></div>
        </div>
      </div>
    `).join('');

    return `
      <div class="stage-detail-card ${i === state.selectedStage ? 'active' : ''}"
           onclick="selectStage(${i})">
        <div class="stage-name">${s.name}</div>
        <div class="stage-range mono">${s.tokenRange}</div>
        <div class="stage-mini-bars">${miniBars}</div>
      </div>
    `;
  }).join('');

  renderStageExpanded(state.selectedStage);
}

function selectStage(idx) {
  state.selectedStage = idx;
  $$('.stage-detail-card').forEach((c, i) => {
    c.classList.toggle('active', i === idx);
  });
  renderStageExpanded(idx);
}

function renderStageExpanded(idx) {
  const panel = $('#stageExpandedDetail');
  if (!panel) return;
  const s = DATA.curriculumStages[idx];
  const keys = ['web','code','indic','stem','reasoning','longctx','agentic'];
  const stageColors = ['#6366f1','#22d3ee','#f59e0b','#10b981','#8b5cf6'];

  const bars = keys.map(k => `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
      <div style="width:80px;font-size:11px;color:var(--text-secondary);">${k}</div>
      <div style="flex:1;height:12px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:${s[k]||0}%;background:${STAGE_COLORS[k]};border-radius:4px;transition:width 0.6s ease;"></div>
      </div>
      <div class="mono" style="font-size:11px;width:32px;text-align:right;color:${STAGE_COLORS[k]};">${s[k]||0}%</div>
    </div>
  `).join('');

  panel.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid ${stageColors[idx]}40;border-radius:var(--radius-md);padding:24px;animation:fadeSlideIn 0.2s ease;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-size:18px;font-weight:700;">${s.name} Stage</div>
          <div class="mono" style="font-size:12px;color:var(--text-muted);">${s.tokenRange}</div>
        </div>
        <span class="badge badge-purple">${s.bands}</span>
      </div>
      ${bars}
      <div style="margin-top:14px;padding:10px;background:var(--bg-card-alt);border-radius:6px;font-size:12px;color:var(--text-secondary);">${s.note}</div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════
// SECTION 8 — Proxy Experiment
// ═══════════════════════════════════════════════════════════

function renderProxy() {
  // Variant cards
  const vc = $('#variantCards');
  if (vc) {
    vc.innerHTML = DATA.proxyExperiment.scale1B.variants.map(v => `
      <div class="variant-card">
        <div class="variant-id ${v.id.toLowerCase()}">${v.id}</div>
        <div class="variant-label">${v.label}</div>
        <div class="variant-desc mono" style="font-size:10px;">${v.description}</div>
      </div>
    `).join('');
  }

  // Metrics table
  const mt = $('#metricTbody');
  if (mt) {
    mt.innerHTML = DATA.proxyExperiment.metrics.map(m => `
      <tr>
        <td style="font-weight:600;color:var(--text-primary);">${m.metric}</td>
        <td style="color:var(--text-secondary);">${m.benchmark}</td>
        <td>${m.decision}</td>
      </tr>
    `).join('');
  }

  // Gating items
  const gi = $('#gatingItems');
  if (gi) {
    const items = [
      { label: 'General Web',   req: '≥ 100B cleaned, deduplicated, manifested tokens' },
      { label: 'Indic Tier A',  req: '≥ 30B verified tokens with educational score ≥ 2.0' },
      { label: 'Code',          req: '≥ 50B Stack v2 tokens through all 8 pipeline stages' },
      { label: 'Reasoning',     req: '≥ 20B AON tokens with pipeline stages 1–7 applied' },
      { label: 'All shards',    req: 'Valid provenance manifest (SHA-256, shard_id, lang_dist)' },
    ];
    gi.innerHTML = items.map(it => `
      <div style="display:flex;align-items:flex-start;gap:10px;padding:8px;background:var(--bg-card-alt);border-radius:var(--radius-sm);">
        <span style="color:var(--rose);flex-shrink:0;margin-top:2px;">⛔</span>
        <div>
          <div style="font-size:12px;font-weight:600;">${it.label}</div>
          <div style="font-size:11px;color:var(--text-muted);">${it.req}</div>
        </div>
      </div>
    `).join('');
  }
}

// ═══════════════════════════════════════════════════════════
// NAV — Active section on scroll
// ═══════════════════════════════════════════════════════════

function initNav() {
  const navItems = $$('.nav-item[data-target]');
  const sections = ['mixture','indic','floors','anneal','agentic','bands','curriculum','proxy']
    .map(id => document.getElementById(id));

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

// ═══════════════════════════════════════════════════════════
// Mixture tab toggle
// ═══════════════════════════════════════════════════════════

function initTabs() {
  const tp = $('#tabPretrain');
  const ta = $('#tabAnneal');
  if (!tp || !ta) return;

  tp.addEventListener('click', () => {
    state.mixMode = 'pretrain';
    state.selectedLane = null;
    tp.classList.add('active');
    ta.classList.remove('active');
    renderMixture();
  });
  ta.addEventListener('click', () => {
    state.mixMode = 'anneal';
    state.selectedLane = null;
    ta.classList.add('active');
    tp.classList.remove('active');
    renderMixture();
  });
}

// ═══════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  renderMixture();
  renderIndicTiers();
  renderFloors();
  renderAnneal();
  renderAgentic();
  renderBands();
  renderCurriculum();
  renderProxy();
  initTabs();
  initNav();
});
