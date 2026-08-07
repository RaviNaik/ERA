/**
 * app.js - TDES Dashboard Application
 * Loads all generated artifacts and renders the interactive dashboard.
 *
 * Data comes from window.TDES_DATA, which is defined in data.js — a file
 * written by src/audit/webapp_export.py at the end of every `run_demo.py`
 * run. Nothing below is hand-typed: every number is read back off the real
 * submission_artifacts/ bundle produced by the Python project. Re-run the
 * demo (`uv run python run_demo.py` inside train_data_exec_system/) and
 * refresh this page to see the new run reflected here.
 */

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
const OPUS_SUMMARY = D.opusSummary || {};
const TOKENIZER = D.tokenizer || {};
const MIXTURE_ACTUAL = D.mixtureActual || {};

if (!HAS_DATA) {
  console.warn('[TDES Dashboard] No data.js found — run `uv run python run_demo.py` ' +
    'inside train_data_exec_system/ to generate submission_artifacts/ and webapp/data.js.');
}

// The configured `total_steps` is a target, not a hard ceiling — the resumed
// run continues until the dataloader is exhausted, so real step ids (and
// checkpoint steps) can run past it. Charts should size their axis to what
// actually happened, not what was originally requested.
function maxAxisStep() {
  const stepNums = [
    META.total_steps || 0,
    ...LEARNING.map(l => l.step),
    ...CHECKPOINTS.map(c => c.step),
  ];
  return Math.max(0, ...stepNums);
}

// ── Lane Colors ────────────────────────────────────────────────────────────
const LANE_COLORS = {
  web: '#3b82f6', code: '#10b981', indic: '#f59e0b',
  agentic: '#a855f7', eval: '#ef4444', reasoning: '#22d3ee',
  stem: '#f97316', long_context: '#e879f9', mixed: '#6b7280'
};
const LANE_ICONS = { web: '🌐', code: '💻', indic: '🇮🇳', agentic: '🤖', eval: '🔒', reasoning: '🧮', stem: '🔬' };

// ── Render: Shards ─────────────────────────────────────────────────────────
function renderShards() {
  const grid = document.getElementById('shard-grid');
  MANIFESTS.forEach(m => {
    const card = document.createElement('div');
    const isBlocked = m.admission !== 'admitted';
    card.className = `shard-card ${isBlocked ? 'blocked' : ''} ${m.is_eval ? 'eval' : ''}`;
    const laneColor = LANE_COLORS[m.lane] || '#6b7280';
    const laneIcon = LANE_ICONS[m.lane] || '📄';
    card.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span class="shard-lane-dot" style="background:${laneColor}"></span>
        <span style="font-size:0.82rem;font-weight:600;color:#fff">${laneIcon} ${m.lane}${m.is_eval ? ' (EVAL)' : ''}</span>
        ${isBlocked ? '<span class="fail-chip pass-chip" style="margin-left:auto">BLOCKED</span>' : '<span class="pass-chip" style="margin-left:auto">ADMITTED</span>'}
      </div>
      <div class="shard-id">${m.shard_id}</div>
      <div class="shard-meta">${m.token_count} tokens · lang: ${m.language}</div>
    `;
    grid.appendChild(card);
  });
}

// ── Render: Packing ────────────────────────────────────────────────────────
function renderPacking() {
  const grid = document.getElementById('packing-grid');
  const policyInfo = {
    pad_each_doc:         { risk: 'None',   riskClass: 'risk-none', desc: 'Simple but wastes compute. Safe boundaries.' },
    concat_and_chop:      { risk: 'High',   riskClass: 'risk-high', desc: 'Pretraining only. Cross-document leakage risk.' },
    greedy_pack:          { risk: 'Medium', riskClass: 'risk-med',  desc: 'Efficient. EOS boundaries prevent leakage.' },
    best_fit_pack:        { risk: 'Medium', riskClass: 'risk-med',  desc: 'Bin-packing variant. Same efficiency as greedy.' },
    structure_preserving: { risk: 'Low',    riskClass: 'risk-low',  desc: 'Per-doc attention masks. Required for SFT/agentic.' },
  };
  Object.entries(PACKING_REPORT).forEach(([policy, stats]) => {
    const info = policyInfo[policy] || {};
    const isBest = policy === 'structure_preserving';
    const card = document.createElement('div');
    card.className = `packing-card ${isBest ? 'best' : ''}`;
    const color = stats.utilization_pct > 90 ? '#10b981' : stats.utilization_pct > 75 ? '#6366f1' : '#f59e0b';
    card.innerHTML = `
      <div class="pack-policy">${policy.replace(/_/g,' ')}</div>
      <div class="util-pct">${stats.utilization_pct}%</div>
      <div class="util-bar-bg">
        <div class="util-bar-fill" style="width:${stats.utilization_pct}%;background:${color}"></div>
      </div>
      <div class="shard-meta">${stats.num_sequences} sequences</div>
      <div class="pack-risk ${info.riskClass}">Boundary risk: ${info.risk}</div>
      <div class="shard-meta" style="margin-top:6px">${info.desc}</div>
      ${isBest ? '<div class="pack-best-tag">★ PRODUCTION POLICY</div>' : ''}
    `;
    grid.appendChild(card);
  });
}

// ── Render: Curriculum Stages ──────────────────────────────────────────────
function renderCurriculum() {
  const container = document.getElementById('curriculum-stages');
  CURRICULUM_STAGES.forEach((stage, si) => {
    const card = document.createElement('div');
    card.className = 'stage-card';
    const lanesHtml = Object.entries(stage.weights).map(([lane, w]) => {
      const pct = Math.round(w * 100);
      const color = LANE_COLORS[lane] || '#6b7280';
      return `<div class="stage-lane">
        <span style="width:80px;font-size:0.75rem;color:var(--text-muted)">${lane}</span>
        <div class="stage-lane-bar"><div class="stage-lane-fill" style="width:${pct*3}px;max-width:100%;background:${color}"></div></div>
        <span class="stage-lane-pct">${pct}%</span>
      </div>`;
    }).join('');
    card.innerHTML = `
      <div class="stage-name">Stage ${si+1}: ${stage.name}</div>
      <div class="stage-budget">Token budget: ${stage.budget}</div>
      <div class="stage-lanes">${lanesHtml}</div>
    `;
    container.appendChild(card);
  });
}

// ── Render: OPUS Decisions ─────────────────────────────────────────────────
function renderOPUS() {
  const accepted = OPUS_DECISIONS.filter(d => d.decision === 'accept').length;
  const rejected = OPUS_DECISIONS.filter(d => d.decision === 'reject').length;
  const deferred = OPUS_DECISIONS.filter(d => d.decision === 'defer').length;
  const overrides = OPUS_DECISIONS.filter(d => d.reason.includes('floor_override')).length;

  document.getElementById('opus-accepted').textContent = accepted;
  document.getElementById('opus-rejected').textContent = rejected;
  document.getElementById('opus-deferred').textContent = deferred;
  document.getElementById('opus-overrides').textContent = overrides;

  const tbody = document.getElementById('opus-tbody');
  OPUS_DECISIONS.forEach(d => {
    const tr = document.createElement('tr');
    const decClass = { accept: 'badge-accept', reject: 'badge-reject', defer: 'badge-defer' }[d.decision] || '';
    const isOverride = d.reason.includes('floor_override');
    tr.innerHTML = `
      <td>${d.step}</td>
      <td class="mono" style="font-size:0.72rem">${d.batch_id}</td>
      <td><span style="color:${LANE_COLORS[d.lane]}">${LANE_ICONS[d.lane]} ${d.lane}</span></td>
      <td>${d.score.toFixed(4)}</td>
      <td><span class="${decClass}">${d.decision.toUpperCase()}</span>${isOverride ? '<span class="badge-override ml-8">⚡ FLOOR</span>' : ''}</td>
      <td style="font-size:0.72rem;color:var(--text-muted)">${d.reason}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: Loss Chart ─────────────────────────────────────────────────────
function renderLossChart() {
  const canvas = document.getElementById('loss-chart');
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

  // Background gradient
  const grd = ctx.createLinearGradient(0, padT, 0, H - padB);
  grd.addColorStop(0, 'rgba(99,102,241,0.15)');
  grd.addColorStop(1, 'rgba(99,102,241,0)');

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const val = maxLoss - (i / 4) * (maxLoss - minLoss);
    ctx.fillStyle = 'rgba(122,127,149,0.8)';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(val.toFixed(3), 4, y + 4);
  }

  // Crash and checkpoint markers (from the actual run config, not guessed)
  const crashStep = META.crash_at_step;
  const ckptSteps = [];
  for (let s = META.checkpoint_every; s <= axisMax; s += META.checkpoint_every) ckptSteps.push(s);
  // Crash line
  ctx.strokeStyle = 'rgba(239,68,68,0.7)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.beginPath(); ctx.moveTo(xScale(crashStep), padT); ctx.lineTo(xScale(crashStep), H - padB); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ef4444'; ctx.font = 'bold 11px Inter';
  ctx.fillText('CRASH', xScale(crashStep) + 4, padT + 16);

  // Checkpoint lines
  ckptSteps.forEach(step => {
    ctx.strokeStyle = 'rgba(99,102,241,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xScale(step), padT); ctx.lineTo(xScale(step), H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(99,102,241,0.9)'; ctx.font = '10px Inter';
    ctx.fillText('CKPT', xScale(step) + 2, padT + 28);
  });

  // Fill area under line
  ctx.beginPath();
  ctx.moveTo(xScale(steps[0]), yScale(losses[0]));
  steps.forEach((s, i) => ctx.lineTo(xScale(s), yScale(losses[i])));
  ctx.lineTo(xScale(steps[steps.length-1]), H - padB);
  ctx.lineTo(xScale(steps[0]), H - padB);
  ctx.closePath();
  ctx.fillStyle = grd; ctx.fill();

  // Loss line
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round'; ctx.beginPath();
  steps.forEach((s, i) => {
    if (i === 0) ctx.moveTo(xScale(s), yScale(losses[i]));
    else ctx.lineTo(xScale(s), yScale(losses[i]));
  });
  ctx.stroke();

  // Lane color dots on line
  steps.forEach((s, i) => {
    const lane = LEARNING[i].lane;
    ctx.beginPath();
    ctx.arc(xScale(s), yScale(losses[i]), 4, 0, Math.PI * 2);
    ctx.fillStyle = LANE_COLORS[lane] || '#6366f1';
    ctx.fill();
  });

  // X axis labels
  ctx.fillStyle = 'rgba(122,127,149,0.8)'; ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  const labelSteps = [0, ...ckptSteps].filter((v, i, a) => a.indexOf(v) === i);
  labelSteps.forEach(s => {
    ctx.fillText(`S${s}`, xScale(s), H - padB + 16);
  });
  ctx.textAlign = 'left';
}

// ── Render: Crash Timeline ─────────────────────────────────────────────────
function renderTimeline() {
  const track = document.getElementById('timeline-track');
  const TOTAL = maxAxisStep();
  const CRASH = META.crash_at_step;
  const RESUME = META.resume_from_step;
  const REPLAY_START = META.replay_start;
  const REPLAY_END = META.replay_end;
  const CKPT_STEPS = new Set();
  for (let s = META.checkpoint_every; s <= TOTAL; s += META.checkpoint_every) CKPT_STEPS.add(s);
  const REPLAY = new Set();
  for (let s = REPLAY_START; s < REPLAY_END; s++) REPLAY.add(s);

  for (let i = 0; i <= TOTAL; i++) {
    const step = document.createElement('div');
    step.className = 'tl-step';
    let dotClass = '';
    let label = '';
    if (CKPT_STEPS.has(i) && i !== CRASH) { dotClass = 'ckpt'; label = 'CKPT'; }
    if (i === CRASH) { dotClass = 'crash'; label = '💥'; }
    // The checkpoint at RESUME is the one crash recovery actually loads —
    // mark it as the resume point instead of an arbitrary post-crash step.
    if (i === RESUME) { dotClass = 'resume'; label = 'RESUME'; }
    if (REPLAY.has(i)) { dotClass = dotClass || 'replay'; }

    step.innerHTML = `<div class="tl-dot ${dotClass}" title="step ${i}"></div>
      <div class="tl-step-num">${i}</div>
      <div class="tl-step-label">${label}</div>`;
    track.appendChild(step);

    if (i < TOTAL) {
      const line = document.createElement('div');
      line.className = `tl-line ${i < CRASH ? 'done' : i >= CRASH && i < RESUME ? 'crashed' : 'resumed'}`;
      track.appendChild(line);
    }
  }
}

// ── Render: Consumption Ledger Table ──────────────────────────────────────
function renderConsumption() {
  const tbody = document.getElementById('consumption-tbody');
  CONSUMPTION.forEach(e => {
    const tr = document.createElement('tr');
    const util = (e.utilization * 100).toFixed(1);
    tr.innerHTML = `
      <td>${e.step}</td>
      <td class="mono" style="font-size:0.72rem">${e.batch_id}</td>
      <td><span style="color:${LANE_COLORS[e.lane]}">${LANE_ICONS[e.lane] || ''} ${e.lane}</span></td>
      <td>${e.useful_tokens}</td>
      <td>${e.loss_bearing_tokens}</td>
      <td><div style="display:flex;align-items:center;gap:8px"><div style="flex:1;height:6px;background:var(--bg-base);border-radius:3px;overflow:hidden"><div style="width:${util}%;height:100%;background:var(--accent);border-radius:3px"></div></div>${util}%</div></td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: Learning Ledger Table ─────────────────────────────────────────
function renderLearning() {
  const tbody = document.getElementById('learning-tbody');
  LEARNING.forEach(e => {
    const tr = document.createElement('tr');
    const docCount = Object.keys(e.doc_loss_map).length;
    tr.innerHTML = `
      <td>${e.step}</td>
      <td class="mono" style="font-size:0.72rem">${e.batch_id}</td>
      <td><span style="color:${LANE_COLORS[e.lane]}">${LANE_ICONS[e.lane] || ''} ${e.lane}</span></td>
      <td><span style="color:var(--accent-amber)">${e.loss.toFixed(4)}</span></td>
      <td>${e.perplexity.toLocaleString()}</td>
      <td>${e.tokens_per_sec}</td>
      <td><span class="pass-chip">${docCount} docs</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: Checkpoints ────────────────────────────────────────────────────
function renderCheckpoints() {
  const grid = document.getElementById('checkpoint-grid');
  CHECKPOINTS.forEach(c => {
    const isFork = !!c.forked_from;
    const card = document.createElement('div');
    card.className = 'ckpt-card';
    card.innerHTML = `
      <div class="ckpt-id">${c.checkpoint_id}</div>
      <div style="margin-bottom:8px"><span class="ckpt-type-badge ${isFork ? 'ckpt-fork' : 'ckpt-main'}">${isFork ? '🍴 FORK' : '💾 MAIN'}</span></div>
      <div class="ckpt-meta">
        Step: <strong>${c.step}</strong><br/>
        Ledger offset: <strong>${c.ledger_offset}</strong><br/>
        Branch: <strong>${c.branch_id}</strong><br/>
        Weights: <span class="mono" style="font-size:0.7rem">${c.weights_hash}</span>
        ${isFork ? `<br/>Forked from: <span class="mono" style="font-size:0.68rem;color:var(--accent-purple)">${c.forked_from}</span>` : ''}
      </div>
    `;
    grid.appendChild(card);
  });
}

// ── Render: Replay Table ───────────────────────────────────────────────────
function renderReplay() {
  const tbody = document.getElementById('replay-tbody');
  REPLAY_HASHES.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.step}</td>
      <td class="mono" style="font-size:0.72rem">${r.original_batch_id}</td>
      <td class="mono" style="font-size:0.72rem;color:var(--accent)">${r.original_hash}</td>
      <td class="mono" style="font-size:0.72rem;color:var(--accent-green)">${r.replay_hash}</td>
      <td><span class="${r.match ? 'match-yes' : 'match-no'}">${r.match ? '✓ MATCH' : '✗ MISMATCH'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render: Evidence Board ─────────────────────────────────────────────────
function renderEvidence() {
  const grid = document.getElementById('evidence-grid');
  const evInfo = {
    tokenizer_integrity: { icon: '🔑', label: 'Tokenizer Integrity' },
    eval_firewall:       { icon: '🛡️', label: 'Evaluation Firewall' },
    packing_correctness: { icon: '📦', label: 'Packing Correctness' },
    crash_recovery:      { icon: '🔄', label: 'Crash Recovery' },
    replay:              { icon: '⏪', label: 'Replay Hash Match' },
    opus_audit_trail:    { icon: '📊', label: 'OPUS Audit Trail' },
    mixture_compliance:  { icon: '⚖️', label: 'Mixture Compliance' },
    learning_trace:      { icon: '📈', label: 'Learning Trace' },
    throughput:          { icon: '⚡', label: 'Throughput & Efficiency' },
  };

  Object.entries(EVIDENCE).forEach(([key, val]) => {
    const info = evInfo[key] || { icon: '✅', label: key };
    const isPass = val.result === 'PASS';
    const card = document.createElement('div');
    card.className = `evidence-card ${isPass ? 'pass' : 'fail'}`;
    card.innerHTML = `
      <div class="ev-header">
        <span class="ev-icon">${info.icon}</span>
        <div style="flex:1">
          <div class="ev-title">${info.label}</div>
          <div class="ev-result ${isPass ? 'pass' : 'fail'}">${val.result}</div>
        </div>
      </div>
      <div class="ev-desc">${val.description}</div>
      <div class="ev-evidence">Evidence: ${val.evidence}</div>
    `;
    grid.appendChild(card);
  });

  // Score breakdown — each row's pass/fail is derived from the evidence.json
  // flags for the requirement(s) it covers, not asserted.
  const passed = key => (EVIDENCE[key] || {}).result === 'PASS';
  const allPass = Object.keys(EVIDENCE).length > 0 && Object.values(EVIDENCE).every(v => v.result === 'PASS');
  const breakdown = [
    ['End-to-end execution', 150, allPass],
    ['Shards + Manifests + Tokenizer', 100, passed('tokenizer_integrity')],
    ['Packing + Masks', 150, passed('packing_correctness')],
    ['Mixture + OPUS + Floors', 150, passed('mixture_compliance') && passed('opus_audit_trail')],
    ['Consumption + Learning Ledgers', 150, passed('learning_trace')],
    ['Checkpoint + Crash + Resume + Replay + Fork', 150, passed('crash_recovery') && passed('replay')],
    ['Eval Firewall', 50, passed('eval_firewall')],
    ['Throughput + Efficiency', 50, passed('throughput')],
    ['Tests + Evidence + Docs', 50, Object.keys(EVIDENCE).length > 0],
  ];
  const breakdownEl = document.getElementById('score-breakdown');
  let achieved = 0, total = 0;
  breakdown.forEach(([area, pts, ok]) => {
    total += pts;
    if (ok) achieved += pts;
    const row = document.createElement('div');
    row.className = 'sb-row';
    row.innerHTML = `<span class="sb-area">${ok ? '✅' : '⬜'} ${area}</span><span class="sb-pts">${ok ? pts : 0} / ${pts} pts</span>`;
    breakdownEl.appendChild(row);
  });

  const scoreNum = document.querySelector('.score-num');
  if (scoreNum) scoreNum.textContent = achieved.toLocaleString();
  const scoreLabel = document.querySelector('.score-label');
  if (scoreLabel) scoreLabel.textContent = `/ ${total.toLocaleString()} pts`;
}

// ── Render: Tokenizer Box ──────────────────────────────────────────────────
function renderTokenizer() {
  const el = id => document.getElementById(id);
  if (TOKENIZER.model_name) el('tok-model').textContent = TOKENIZER.model_name;
  if (TOKENIZER.tokenizer_sha) el('tok-sha').textContent = TOKENIZER.tokenizer_sha;
  if (TOKENIZER.vocab_size) el('tok-vocab').textContent = TOKENIZER.vocab_size.toLocaleString();
}

// ── Render: Protected Floor Compliance ─────────────────────────────────────
function renderFloors() {
  const fc = MIXTURE_ACTUAL.floor_compliance || {};
  const setFloor = (lane, barId, pctId, chipId) => {
    const c = fc[lane];
    if (!c) return;
    const pct = (c.actual * 100).toFixed(1);
    document.getElementById(barId).style.width = Math.min(100, c.actual * 100).toFixed(1) + '%';
    document.getElementById(pctId).textContent = pct + '%';
    const chip = document.getElementById(chipId);
    chip.textContent = c.compliant ? 'COMPLIANT ✓' : 'BELOW FLOOR ✗';
    chip.className = c.compliant ? 'pass-chip' : 'fail-chip';
  };
  setFloor('indic', 'indic-bar', 'indic-pct', 'indic-compliant');
  setFloor('agentic', 'agentic-bar', 'agentic-pct', 'agentic-compliant');
}

// ── Render: Run-config-dependent copy (crash/resume/replay step numbers) ──
function renderRunConfigCopy() {
  const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setText('training-steps-desc', META.total_steps);
  setText('crash-step-desc', META.crash_at_step);
  setText('resume-step-desc', META.resume_from_step);
  setText('crash-step-desc-2', META.crash_at_step);
  setText('resume-step-desc-2', META.resume_from_step);
  setText('replay-range-desc', `${META.replay_start}–${META.replay_end}`);
  setText('fw-count', META.blocked_docs);
  setText('pipe-doc-count', `${META.total_docs ?? '—'} docs · 4 lanes`);
  const passCount = Object.values(EVIDENCE).filter(v => v.result === 'PASS').length;
  const totalCount = Object.keys(EVIDENCE).length;
  setText('pipe-audit-count', `${passCount}/${totalCount || 9} PASS`);
}

// ── Tab switching ──────────────────────────────────────────────────────────
function showTab(name) {
  document.getElementById('panel-consume').classList.toggle('hidden', name !== 'consume');
  document.getElementById('panel-learn').classList.toggle('hidden', name !== 'learn');
  document.getElementById('tab-consume').classList.toggle('active', name === 'consume');
  document.getElementById('tab-learn').classList.toggle('active', name === 'learn');
}

// ── Intersection Observer for Nav ─────────────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id);
      });
    }
  });
}, { threshold: 0.3 });
sections.forEach(s => observer.observe(s));

// ── Pipeline node clicks ───────────────────────────────────────────────────
document.querySelectorAll('.pipe-node').forEach(node => {
  node.addEventListener('click', () => {
    const section = node.dataset.section;
    document.getElementById(section)?.scrollIntoView({ behavior: 'smooth' });
  });
});

// ── Animate stat numbers ───────────────────────────────────────────────────
function animateNum(el, target, duration = 1200, isFloat = false) {
  const start = parseFloat(el.textContent) || 0;
  const startTime = performance.now();
  function update(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = start + (target - start) * eased;
    el.textContent = isFloat ? val.toFixed(1) + '%' : Math.round(val).toLocaleString();
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── INIT ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  if (!HAS_DATA) {
    const banner = document.createElement('div');
    banner.style.cssText = 'position:sticky;top:0;z-index:999;background:#7c2d12;color:#fff;' +
      'padding:10px 20px;text-align:center;font:600 0.85rem Inter,sans-serif;';
    banner.textContent = '⚠ No run data found (webapp/data.js is missing). Run `uv run python run_demo.py` ' +
      'inside train_data_exec_system/ to generate it, then reload this page.';
    document.body.prepend(banner);
  }

  renderTokenizer();
  renderShards();
  renderPacking();
  renderCurriculum();
  renderFloors();
  renderOPUS();
  renderConsumption();
  renderLearning();
  renderCheckpoints();
  renderReplay();
  renderEvidence();
  renderTimeline();
  renderRunConfigCopy();

  // Render chart with slight delay
  setTimeout(renderLossChart, 100);

  // Animate hero stats (real numbers from this run)
  setTimeout(() => {
    animateNum(document.getElementById('stat-shards'), META.total_docs || 0);
    animateNum(document.getElementById('stat-admitted'), META.admitted_docs || 0);
    animateNum(document.getElementById('stat-steps'), PERFORMANCE.total_steps || 0);
    animateNum(document.getElementById('stat-util'), PERFORMANCE.packing_utilization_pct || 0, 1500, true);
    animateNum(document.getElementById('stat-tests'), META.test_count || 0);
  }, 300);

  // Update metrics
  document.getElementById('tm-avg-loss').textContent = (PERFORMANCE.average_loss ?? 0).toFixed(4);
  document.getElementById('tm-tps').textContent = Math.round(PERFORMANCE.average_tokens_per_sec || 0);
  document.getElementById('tm-steps').textContent = PERFORMANCE.total_steps ?? '—';
  document.getElementById('tm-util').textContent = (PERFORMANCE.packing_utilization_pct ?? 0).toFixed(1) + '%';

  const allPass = Object.keys(EVIDENCE).length > 0 && Object.values(EVIDENCE).every(v => v.result === 'PASS');
  const badge = document.getElementById('all-pass-badge');
  if (badge && !allPass) {
    badge.innerHTML = '<span class="badge-dot"></span> CHECK EVIDENCE';
    badge.style.background = 'rgba(239,68,68,0.15)';
    badge.style.color = '#ef4444';
  }

  console.log(`[TDES Dashboard] Loaded from data.js generated ${META.generated_at || 'unknown time'}. ` +
    `${Object.values(EVIDENCE).filter(v => v.result === 'PASS').length}/${Object.keys(EVIDENCE).length} requirements PASS.`);
});
