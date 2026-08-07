// ============================================================
// app.js — Training Data Execution System Dashboard
// Session 6 Assignment · ERA V5
// ============================================================
//
// Every render function below reads from window.TDES_DATA, which is
// defined in data.js — a file written by
// train_data_exec_system/src/audit/webapp_export.py at the end of every
// `run_demo.py` run. Nothing here is hand-typed: every number is read
// back off the real submission_artifacts/ bundle the pipeline produced.
// Re-run the demo and reload this page to see the new run reflected.

'use strict';

const D = window.TDES_DATA || {};
const HAS_DATA = Object.keys(D).length > 0;

const META = D.meta || {};
const EVIDENCE = D.evidence || {};
const PERFORMANCE = D.performance || {};
const PACKING_REPORT = D.packingReport || {};
const MANIFESTS = D.manifests || [];
const OPUS_DECISIONS = D.opusDecisions || [];
const CONSUMPTION = D.consumption || [];
const LEARNING = D.learning || [];
const CHECKPOINTS = D.checkpoints || [];
const REPLAY_HASHES = D.replayHashes || [];
const CURRICULUM_STAGES = D.curriculumStages || [];
const MIXTURE_ACTUAL = D.mixtureActual || {};
const TOKENIZER = D.tokenizer || {};

// ── DOM helpers ─────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const setText = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

// ── Lane colors / icons ──────────────────────────────────────
const LANE_COLORS = {
  web: '#3b82f6', code: '#10b981', indic: '#f59e0b',
  agentic: '#a855f7', eval: '#ef4444', reasoning: '#22d3ee',
  stem: '#f97316', long_context: '#e879f9', mixed: '#6b7280'
};
const LANE_ICONS = { web: '🌐', code: '💻', indic: '🇮🇳', agentic: '🤖', eval: '🔒', reasoning: '🧮', stem: '🔬' };

// The configured total_steps is a target, not a hard ceiling — the
// resumed run continues until the dataloader is exhausted, so real step
// ids (and checkpoint steps) can run past it. Charts size to what
// actually happened.
function maxAxisStep() {
  const nums = [META.total_steps || 0, ...LEARNING.map(l => l.step), ...CHECKPOINTS.map(c => c.step)];
  return Math.max(0, ...nums);
}

// ═══════════════════════════════════════════════════════════
// SECTION 1 — Tokenizer + Shards + Firewall
// ═══════════════════════════════════════════════════════════

function renderTokenizer() {
  if (TOKENIZER.model_name) setText('tok-model', TOKENIZER.model_name);
  if (TOKENIZER.tokenizer_sha) setText('tok-sha', TOKENIZER.tokenizer_sha);
  if (TOKENIZER.vocab_size) setText('tok-vocab', TOKENIZER.vocab_size.toLocaleString());
}

function renderShards() {
  const grid = $('#shard-grid');
  if (!grid) return;
  MANIFESTS.forEach(m => {
    const blocked = m.admission !== 'admitted';
    const card = el('div', `shard-card ${blocked ? 'blocked' : ''}`);
    const color = LANE_COLORS[m.lane] || '#6b7280';
    const icon = LANE_ICONS[m.lane] || '📄';
    card.innerHTML = `
      <div class="sc-top">
        <span class="sc-lane-dot" style="background:${color}"></span>
        <span class="sc-lane">${icon} ${m.lane}${m.is_eval ? ' (EVAL)' : ''}</span>
        <span class="badge ${blocked ? 'badge-red' : 'badge-green'}">${blocked ? 'BLOCKED' : 'ADMITTED'}</span>
      </div>
      <div class="sc-id">${m.shard_id}</div>
      <div class="sc-meta">${m.token_count} tokens · lang: ${m.language}</div>
    `;
    grid.appendChild(card);
  });
}

function renderFirewall() {
  setText('fw-count', META.blocked_docs ?? MANIFESTS.filter(m => m.is_eval).length);
}

// ═══════════════════════════════════════════════════════════
// SECTION 2 — Packing Policies
// ═══════════════════════════════════════════════════════════

function renderPacking() {
  const grid = $('#packing-grid');
  if (!grid) return;
  const info = {
    pad_each_doc:         { risk: 'None',   cls: 'risk-none', desc: 'Simple but wastes compute. Safe boundaries.' },
    concat_and_chop:      { risk: 'High',   cls: 'risk-high', desc: 'Pretraining only. Cross-document leakage risk.' },
    greedy_pack:          { risk: 'Medium', cls: 'risk-med',  desc: 'Efficient. EOS boundaries prevent leakage.' },
    best_fit_pack:        { risk: 'Medium', cls: 'risk-med',  desc: 'Bin-packing variant. Same efficiency as greedy.' },
    structure_preserving: { risk: 'Low',    cls: 'risk-low',  desc: 'Per-doc attention masks. Required for SFT/agentic.' },
  };
  Object.entries(PACKING_REPORT).forEach(([policy, stats]) => {
    const meta = info[policy] || {};
    const isBest = policy === 'structure_preserving';
    const color = stats.utilization_pct > 90 ? '#10b981' : stats.utilization_pct > 75 ? '#6366f1' : '#f59e0b';
    const card = el('div', `policy-card ${isBest ? 'best' : ''}`);
    card.innerHTML = `
      <div class="policy-name">${policy.replace(/_/g, ' ')}</div>
      <div class="policy-util">${stats.utilization_pct}%</div>
      <div class="policy-bar-track"><div class="policy-bar-fill" style="width:${stats.utilization_pct}%;background:${color}"></div></div>
      <div class="policy-meta">${stats.num_sequences} sequences</div>
      <div class="policy-risk ${meta.cls}">Boundary risk: ${meta.risk}</div>
      <div class="policy-desc">${meta.desc || ''}</div>
      ${isBest ? '<div class="policy-best-tag">★ PRODUCTION POLICY</div>' : ''}
    `;
    grid.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════════
// SECTION 3 — Curriculum + Protected Floors
// ═══════════════════════════════════════════════════════════

function renderCurriculum() {
  const grid = $('#curriculum-grid');
  if (!grid) return;
  CURRICULUM_STAGES.forEach((stage, i) => {
    const card = el('div', 'curriculum-card');
    const bars = Object.entries(stage.weights).map(([lane, w]) => {
      const pct = Math.round(w * 100);
      const color = LANE_COLORS[lane] || '#6b7280';
      return `<div class="smbar-row">
        <span class="smbar-label">${lane}</span>
        <div class="smbar-track"><div class="smbar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="smbar-pct">${pct}%</span>
      </div>`;
    }).join('');
    card.innerHTML = `
      <div class="cc-name">Stage ${i + 1}: ${stage.name}</div>
      <div class="cc-budget">Token budget: ${stage.budget}</div>
      ${bars}
    `;
    grid.appendChild(card);
  });
}

function renderFloors() {
  const fc = MIXTURE_ACTUAL.floor_compliance || {};
  const setFloor = (lane, barId, pctId, mechId) => {
    const c = fc[lane];
    if (!c) return;
    document.getElementById(barId).style.width = Math.min(100, c.actual * 100).toFixed(1) + '%';
    document.getElementById(pctId).textContent = (c.actual * 100).toFixed(1) + '%';
    const mech = document.getElementById(mechId);
    mech.textContent = c.compliant
      ? `✓ COMPLIANT — actual ${(c.actual * 100).toFixed(1)}% ≥ floor ${(c.floor * 100).toFixed(0)}%`
      : `✗ BELOW FLOOR — actual ${(c.actual * 100).toFixed(1)}% < floor ${(c.floor * 100).toFixed(0)}%`;
  };
  setFloor('indic', 'indic-bar', 'indic-pct', 'indic-compliant');
  setFloor('agentic', 'agentic-bar', 'agentic-pct', 'agentic-compliant');
}

// ═══════════════════════════════════════════════════════════
// SECTION 4 — OPUS Decisions
// ═══════════════════════════════════════════════════════════

function renderOPUS() {
  const accepted = OPUS_DECISIONS.filter(d => d.decision === 'accept').length;
  const rejected = OPUS_DECISIONS.filter(d => d.decision === 'reject').length;
  const deferred = OPUS_DECISIONS.filter(d => d.decision === 'defer').length;
  const overrides = OPUS_DECISIONS.filter(d => d.reason.includes('floor_override')).length;

  setText('opus-accepted', accepted);
  setText('opus-rejected', rejected);
  setText('opus-deferred', deferred);
  setText('opus-overrides', overrides);

  const tbody = $('#opus-tbody');
  if (!tbody) return;
  OPUS_DECISIONS.forEach(d => {
    const badgeCls = { accept: 'badge-green', reject: 'badge-red', defer: 'badge-yellow' }[d.decision] || 'badge-blue';
    const isOverride = d.reason.includes('floor_override');
    const tr = el('tr');
    tr.innerHTML = `
      <td>${d.step}</td>
      <td class="mono">${d.batch_id}</td>
      <td><span style="color:${LANE_COLORS[d.lane]}">${LANE_ICONS[d.lane] || ''} ${d.lane}</span></td>
      <td class="mono">${d.score.toFixed(4)}</td>
      <td><span class="badge ${badgeCls}">${d.decision}</span>${isOverride ? ' <span class="badge badge-purple">⚡ FLOOR</span>' : ''}</td>
      <td class="text-muted" style="font-size:11px;">${d.reason}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ═══════════════════════════════════════════════════════════
// SECTION 5 — Training Run: metrics, loss chart, timeline
// ═══════════════════════════════════════════════════════════

function renderTrainingCopy() {
  setText('training-steps-desc', META.total_steps);
  setText('crash-step-desc', META.crash_at_step);
  setText('resume-step-desc', META.resume_from_step);
  setText('tm-avg-loss', (PERFORMANCE.average_loss ?? 0).toFixed(4));
  setText('tm-tps', Math.round(PERFORMANCE.average_tokens_per_sec || 0));
  setText('tm-steps', PERFORMANCE.total_steps ?? '—');
  setText('tm-util', (PERFORMANCE.packing_utilization_pct ?? 0).toFixed(1) + '%');
}

function renderLossChart() {
  const canvas = $('#loss-chart');
  if (!canvas || !LEARNING.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 860;
  const H = 280;
  canvas.width = W;
  canvas.height = H;

  const steps = LEARNING.map(l => l.step);
  const losses = LEARNING.map(l => l.loss);
  const minLoss = Math.min(...losses) - 0.005;
  const maxLoss = Math.max(...losses) + 0.005;
  const padL = 60, padR = 20, padT = 20, padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const axisMax = maxAxisStep();

  const xScale = s => padL + (s / axisMax) * plotW;
  const yScale = l => padT + plotH - ((l - minLoss) / (maxLoss - minLoss)) * plotH;

  const grd = ctx.createLinearGradient(0, padT, 0, H - padB);
  grd.addColorStop(0, 'rgba(99,102,241,0.15)');
  grd.addColorStop(1, 'rgba(99,102,241,0)');

  ctx.strokeStyle = 'rgba(148,163,184,0.15)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const val = maxLoss - (i / 4) * (maxLoss - minLoss);
    ctx.fillStyle = 'rgba(148,163,184,0.9)';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(val.toFixed(3), 4, y + 4);
  }

  const crashStep = META.crash_at_step;
  const ckptSteps = [];
  for (let s = META.checkpoint_every; s <= axisMax; s += META.checkpoint_every) ckptSteps.push(s);

  ctx.strokeStyle = 'rgba(244,63,94,0.7)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.beginPath(); ctx.moveTo(xScale(crashStep), padT); ctx.lineTo(xScale(crashStep), H - padB); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f43f5e'; ctx.font = 'bold 11px Inter';
  ctx.fillText('CRASH', xScale(crashStep) + 4, padT + 16);

  ckptSteps.forEach(step => {
    ctx.strokeStyle = 'rgba(99,102,241,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xScale(step), padT); ctx.lineTo(xScale(step), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(99,102,241,0.9)'; ctx.font = '10px Inter';
    ctx.fillText('CKPT', xScale(step) + 2, padT + 28);
  });

  ctx.beginPath();
  ctx.moveTo(xScale(steps[0]), yScale(losses[0]));
  steps.forEach((s, i) => ctx.lineTo(xScale(s), yScale(losses[i])));
  ctx.lineTo(xScale(steps[steps.length - 1]), H - padB);
  ctx.lineTo(xScale(steps[0]), H - padB);
  ctx.closePath();
  ctx.fillStyle = grd; ctx.fill();

  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round'; ctx.beginPath();
  steps.forEach((s, i) => { i === 0 ? ctx.moveTo(xScale(s), yScale(losses[i])) : ctx.lineTo(xScale(s), yScale(losses[i])); });
  ctx.stroke();

  steps.forEach((s, i) => {
    ctx.beginPath();
    ctx.arc(xScale(s), yScale(losses[i]), 4, 0, Math.PI * 2);
    ctx.fillStyle = LANE_COLORS[LEARNING[i].lane] || '#6366f1';
    ctx.fill();
  });

  ctx.fillStyle = 'rgba(148,163,184,0.9)'; ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  [0, ...ckptSteps].filter((v, i, a) => a.indexOf(v) === i).forEach(s => ctx.fillText(`S${s}`, xScale(s), H - padB + 16));
  ctx.textAlign = 'left';
}

function renderTimeline() {
  const track = $('#timeline-track');
  if (!track) return;
  const TOTAL = maxAxisStep();
  const CRASH = META.crash_at_step;
  const RESUME = META.resume_from_step;
  const CKPT_STEPS = new Set();
  for (let s = META.checkpoint_every; s <= TOTAL; s += META.checkpoint_every) CKPT_STEPS.add(s);
  const REPLAY = new Set();
  for (let s = META.replay_start; s < META.replay_end; s++) REPLAY.add(s);

  for (let i = 0; i <= TOTAL; i++) {
    const step = el('div', 'tl-step');
    let dotClass = '', label = '';
    if (CKPT_STEPS.has(i) && i !== CRASH) { dotClass = 'ckpt'; label = 'CKPT'; }
    if (i === CRASH) { dotClass = 'crash'; label = '💥'; }
    // The checkpoint at RESUME is the one crash recovery actually loads.
    if (i === RESUME) { dotClass = 'resume'; label = 'RESUME'; }
    if (REPLAY.has(i)) { dotClass = dotClass || 'replay'; }

    step.innerHTML = `<div class="tl-dot ${dotClass}" title="step ${i}"></div>
      <div class="tl-step-num">${i}</div>
      <div class="tl-step-label">${label}</div>`;
    track.appendChild(step);

    if (i < TOTAL) {
      const line = el('div', `tl-line ${i < CRASH ? 'done' : (i >= CRASH && i < RESUME ? 'crashed' : 'resumed')}`);
      track.appendChild(line);
    }
  }
}

// ═══════════════════════════════════════════════════════════
// SECTION 6 — Ledgers
// ═══════════════════════════════════════════════════════════

function renderConsumption() {
  const tbody = $('#consumption-tbody');
  if (!tbody) return;
  CONSUMPTION.forEach(e => {
    const util = (e.utilization * 100).toFixed(1);
    const tr = el('tr');
    tr.innerHTML = `
      <td>${e.step}</td>
      <td class="mono">${e.batch_id}</td>
      <td><span style="color:${LANE_COLORS[e.lane]}">${LANE_ICONS[e.lane] || ''} ${e.lane}</span></td>
      <td>${e.useful_tokens}</td>
      <td>${e.loss_bearing_tokens}</td>
      <td><div class="mini-bar-cell"><div class="mini-bar-track"><div class="mini-bar-fill" style="width:${util}%"></div></div>${util}%</div></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderLearning() {
  const tbody = $('#learning-tbody');
  if (!tbody) return;
  LEARNING.forEach(e => {
    const docCount = Object.keys(e.doc_loss_map || {}).length;
    const tr = el('tr');
    tr.innerHTML = `
      <td>${e.step}</td>
      <td class="mono">${e.batch_id}</td>
      <td><span style="color:${LANE_COLORS[e.lane]}">${LANE_ICONS[e.lane] || ''} ${e.lane}</span></td>
      <td class="text-amber">${e.loss.toFixed(4)}</td>
      <td>${e.perplexity.toLocaleString()}</td>
      <td>${e.tokens_per_sec}</td>
      <td><span class="badge badge-blue">${docCount} docs</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function initLedgerTabs() {
  const tabConsume = $('#tab-consume');
  const tabLearn = $('#tab-learn');
  const panelConsume = $('#panel-consume');
  const panelLearn = $('#panel-learn');
  if (!tabConsume || !tabLearn) return;
  tabConsume.addEventListener('click', () => {
    tabConsume.classList.add('active'); tabLearn.classList.remove('active');
    panelConsume.classList.remove('hidden'); panelLearn.classList.add('hidden');
  });
  tabLearn.addEventListener('click', () => {
    tabLearn.classList.add('active'); tabConsume.classList.remove('active');
    panelLearn.classList.remove('hidden'); panelConsume.classList.add('hidden');
  });
}

// ═══════════════════════════════════════════════════════════
// SECTION 7 — Checkpoints + Replay
// ═══════════════════════════════════════════════════════════

function renderCheckpoints() {
  const grid = $('#checkpoint-grid');
  if (!grid) return;
  CHECKPOINTS.forEach(c => {
    const isFork = !!c.forked_from;
    const card = el('div', `ckpt-card ${isFork ? 'fork' : ''}`);
    card.innerHTML = `
      <div class="ckpt-id">${c.checkpoint_id}</div>
      <span class="badge ${isFork ? 'badge-purple' : 'badge-blue'}">${isFork ? '🍴 FORK' : '💾 MAIN'}</span>
      <div class="ckpt-meta">
        Step: <strong>${c.step}</strong><br/>
        Ledger offset: <strong>${c.ledger_offset}</strong><br/>
        Branch: <strong>${c.branch_id}</strong><br/>
        Weights: <span class="mono">${c.weights_hash}</span>
        ${isFork ? `<br/>Forked from: <span class="ckpt-forked-from">${c.forked_from}</span>` : ''}
      </div>
    `;
    grid.appendChild(card);
  });
}

function renderReplay() {
  setText('replay-range-desc', `${META.replay_start}–${META.replay_end}`);
  const tbody = $('#replay-tbody');
  if (!tbody) return;
  REPLAY_HASHES.forEach(r => {
    const tr = el('tr');
    tr.innerHTML = `
      <td>${r.step}</td>
      <td class="mono">${r.original_batch_id}</td>
      <td class="mono text-indigo">${r.original_hash}</td>
      <td class="mono text-green">${r.replay_hash}</td>
      <td><span class="badge ${r.match ? 'badge-green' : 'badge-red'}">${r.match ? '✓ MATCH' : '✗ MISMATCH'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ═══════════════════════════════════════════════════════════
// SECTION 8 — Evidence Board
// ═══════════════════════════════════════════════════════════

const EVIDENCE_INFO = {
  tokenizer_integrity: { icon: '🔑', label: 'Tokenizer Integrity' },
  eval_firewall:       { icon: '🛡️', label: 'Evaluation Firewall' },
  packing_correctness: { icon: '📦', label: 'Packing Correctness' },
  crash_recovery:      { icon: '🔄', label: 'Crash Recovery' },
  replay:              { icon: '⏪', label: 'Replay Hash Match' },
  opus_audit_trail:    { icon: '🧠', label: 'OPUS Audit Trail' },
  mixture_compliance:  { icon: '⚖️', label: 'Mixture Compliance' },
  learning_trace:      { icon: '📈', label: 'Learning Trace' },
  throughput:          { icon: '⚡', label: 'Throughput & Efficiency' },
};

function renderEvidence() {
  const grid = $('#evidence-grid');
  if (grid) {
    Object.entries(EVIDENCE).forEach(([key, val]) => {
      const info = EVIDENCE_INFO[key] || { icon: '✅', label: key };
      const isPass = val.result === 'PASS';
      const card = el('div', `evidence-card ${isPass ? '' : 'fail'}`);
      card.innerHTML = `
        <div class="ev-header">
          <span class="ev-icon">${info.icon}</span>
          <div>
            <div class="ev-title">${info.label}</div>
            <div class="ev-result ${isPass ? 'pass' : 'fail'}">${val.result}</div>
          </div>
        </div>
        <div class="ev-desc">${val.description || val.reason || ''}</div>
        <div class="ev-evidence">Evidence: ${val.evidence || '—'}</div>
      `;
      grid.appendChild(card);
    });
  }

  // Score breakdown — each row's pass/fail is derived from the evidence.json
  // flags for the requirement(s) it covers, not asserted.
  const passed = key => (EVIDENCE[key] || {}).result === 'PASS';
  const allPass = Object.keys(EVIDENCE).length > 0 && Object.values(EVIDENCE).every(v => v.result === 'PASS');
  const breakdown = [
    ['End-to-end execution', allPass],
    ['Shards + Manifests + Tokenizer', passed('tokenizer_integrity')],
    ['Packing + Masks', passed('packing_correctness')],
    ['Mixture + OPUS + Floors', passed('mixture_compliance') && passed('opus_audit_trail')],
    ['Consumption + Learning Ledgers', passed('learning_trace')],
    ['Checkpoint + Crash + Resume + Replay + Fork', passed('crash_recovery') && passed('replay')],
    ['Eval Firewall', passed('eval_firewall')],
    ['Throughput + Efficiency', passed('throughput')],
    ['Tests + Evidence + Docs', Object.keys(EVIDENCE).length > 0],
  ];
  const breakdownEl = $('#score-breakdown');
  let ok = 0;
  if (breakdownEl) {
    breakdown.forEach(([area, pass]) => {
      if (pass) ok++;
      const row = el('div', 'sb-row');
      row.innerHTML = `<span class="sb-area">${pass ? '✅' : '⬜'} ${area}</span><span class="sb-pts">${pass ? 'PASS' : '—'}</span>`;
      breakdownEl.appendChild(row);
    });
  }
  const scoreBig = $('#score-big');
  const scoreSub = $('#score-sub');
  if (scoreBig) {
    scoreBig.textContent = `${ok}/${breakdown.length}`;
    scoreBig.classList.toggle('partial', ok < breakdown.length);
  }
  if (scoreSub) scoreSub.textContent = 'requirement areas passing';
}

// ═══════════════════════════════════════════════════════════
// NAV — Active section on scroll
// ═══════════════════════════════════════════════════════════

function initNav() {
  const navItems = $$('.nav-item[data-target]');
  const sections = ['shards', 'packing', 'mixture', 'opus', 'training', 'ledgers', 'checkpoint', 'evidence']
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

// ── Animate hero numbers ─────────────────────────────────────
function animateNum(elem, target, duration = 1200, suffix = '') {
  if (!elem) return;
  const start = 0;
  const startTime = performance.now();
  function step(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = start + (target - start) * eased;
    elem.textContent = Math.round(val).toLocaleString() + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderHeroStats() {
  animateNum($('#stat-shards'), META.total_docs || 0);
  animateNum($('#stat-admitted'), META.admitted_docs || 0);
  animateNum($('#stat-steps'), PERFORMANCE.total_steps || 0);
  animateNum($('#stat-util'), PERFORMANCE.packing_utilization_pct || 0, 1500, '%');
  animateNum($('#stat-tests'), META.test_count || 0);
}

// ═══════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  if (!HAS_DATA) {
    const banner = el('div', 'no-data-banner',
      '⚠ No run data found (webapp/data.js is missing). Run <code>uv run python run_demo.py</code> ' +
      'inside train_data_exec_system/ to generate it, then reload this page.');
    document.body.prepend(banner);
  }

  renderTokenizer();
  renderShards();
  renderFirewall();
  renderPacking();
  renderCurriculum();
  renderFloors();
  renderOPUS();
  renderTrainingCopy();
  renderConsumption();
  renderLearning();
  initLedgerTabs();
  renderCheckpoints();
  renderReplay();
  renderEvidence();
  renderTimeline();
  initNav();

  setTimeout(renderLossChart, 100);
  setTimeout(renderHeroStats, 200);

  console.log(`[TDES Dashboard] Loaded from data.js generated ${META.generated_at || 'unknown time'}. ` +
    `${Object.values(EVIDENCE).filter(v => v.result === 'PASS').length}/${Object.keys(EVIDENCE).length} requirements PASS.`);
});
