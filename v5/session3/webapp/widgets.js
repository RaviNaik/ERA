// ── widgets.js — render each stage's tab content ─────────────────

// ──────────── STAGE 1: Benchmarks ────────────────────────────────
function renderBenchTab(key) {
  const rows = BENCH[key];
  const tbody = document.getElementById('bench-tbody');
  document.getElementById('bench-chart-title').textContent = BENCH_LABELS[key];
  document.getElementById('bench-insight').innerHTML = '<strong>Insight:</strong> ' + BENCH_INSIGHTS[key];

  tbody.innerHTML = rows.map(r => {
    const diffVal = r.bharat - r.gemma;
    const diffStr = diffVal > 0 ? '+' + diffVal.toFixed(1) : diffVal.toFixed(1);
    const badgeCls = r.gap === '↑' ? 'gap-up' : r.gap === 'close' ? 'gap-close' : 'gap-same';
    const arrow = r.gap === '↑' ? '↑ Exceed' : r.gap === 'close' ? '≈ Close' : '≈ Match';
    return `<tr>
      <td><strong>${r.name}</strong><br><span style="font-size:11px;color:#9b94b0">${r.note}</span></td>
      <td class="num" style="color:#4a90c4">${r.gemma}%</td>
      <td class="num" style="color:#3a9e72">${r.bharat}%</td>
      <td><span class="gap-badge ${badgeCls}">${arrow} ${diffStr}</span></td>
    </tr>`;
  }).join('');

  // Score summary
  const exceed = rows.filter(r => r.gap === '↑').length;
  const close  = rows.filter(r => r.gap === 'close').length;
  const match  = rows.length - exceed - close;
  document.getElementById('sc-match').querySelector('.sc-val').textContent  = match;
  document.getElementById('sc-exceed').querySelector('.sc-val').textContent = exceed;
  document.getElementById('sc-close').querySelector('.sc-val').textContent  = close;

  setTimeout(() => drawBenchChart('bench-chart', rows, BENCH_LABELS[key]), 30);
}

// ──────────── STAGE 2: Data Architecture ─────────────────────────
function renderDataTab(key) {
  const vp = document.getElementById('data-visual-panel');
  const dp = document.getElementById('data-detail-panel');

  if (key === 'tier') {
    vp.innerHTML = `
      <div class="tier-block t1">
        <div class="tier-label">Tier 1 — OPUS-Eligible Pools (D1–D4)</div>
        <div class="tier-desc">Web crawl (FineWeb, Dolma), books, code (The Stack v2), academic (arXiv), India-specific (NCERT). India-perspective classifier retains ~30% of candidates.</div>
        <span class="tier-tag">~13T unique tokens</span>
      </div>
      <div style="text-align:center;color:#9b94b0;font-size:13px;padding:6px 0">↕ OPUS selector picks ~40% per batch</div>
      <div class="tier-block t2">
        <div class="tier-label">Tier 2 — Always-On Indic Channel</div>
        <div class="tier-desc">Sangraha Verified, IndicCorpV2, BPCC, custom .in crawl. <strong>Bypasses English-centric OPUS selector</strong> — guaranteed 8–12% of every batch. Prevents the V4 under-weighting bug.</div>
        <span class="tier-tag">~800B–1T unique → 1.5–2T effective (2× repeat)</span>
      </div>
      <div style="text-align:center;color:#9b94b0;font-size:13px;padding:6px 0">↕ Provides direction only (no training)</div>
      <div class="tier-block t3">
        <div class="tier-label">Tier 3 — Golden Proxy 🔒</div>
        <div class="tier-desc">All benchmark test splits (MMLU, MILU, HumanEval, FLORES). <strong>NEVER trains the model.</strong> n-gram hashed on Day 1; any match in Tier 1/2 is purged.</div>
        <span class="tier-tag">Hard firewall — contamination = test invalidation</span>
      </div>`;
    dp.innerHTML = `
      <div class="info-box"><strong>Why three tiers?</strong> A single OPUS selector trained on English quality signals systematically under-weights Indic text. The always-on Tier 2 guarantees cultural and linguistic coverage, while Tier 3 maintains evaluation integrity.</div>
      ${[
        {l:'Total unique tokens',v:'~15T'},
        {l:'Effective tokens (with repeat)',v:'~18T'},
        {l:'Anneal reserve (held back)',v:'500B unique → 2T eff.'},
        {l:'Indic guaranteed share',v:'8–12% every batch'},
        {l:'India classifier threshold',v:'Retain score ≥ 3'},
        {l:'Contamination check',v:'8-gram exact + Jaccard 0.7'},
      ].map(s=>`<div class="detail-stat"><span class="detail-stat-label">${s.l}</span><span class="detail-stat-val">${s.v}</span></div>`).join('')}`;
  }

  else if (key === 'domain') {
    vp.innerHTML = `<div class="plot-card" style="align-items:center"><h3>Pre-training Domain Mix (% of tokens)</h3>
      <canvas id="domain-donut" width="260" height="260" style="max-width:260px"></canvas>
    </div>`;
    setTimeout(() => drawDomainDonut('domain-donut'), 30);

    dp.innerHTML = `<div class="domain-legend">` +
      DOMAIN_MIX.map(d => `<div class="domain-legend-item">
        <div class="domain-dot" style="background:${d.color}"></div>
        <span>${d.name}</span>
        <span class="domain-legend-val">${d.pct}% · ${d.tokens}</span>
      </div>`).join('') + `</div>
      <div class="info-box" style="margin-top:12px"><strong>Code at 15%:</strong> Empirically, code training improves logical reasoning across all domains — not just coding benchmarks. Math+Science at 8% mirrors the best open training recipes.</div>`;
  }

  else if (key === 'sources') {
    const t1 = KEY_SOURCES.filter(s => s.tier==='T1');
    const t2 = KEY_SOURCES.filter(s => s.tier==='T2');
    const makeRows = arr => arr.map(s => `<tr>
      <td><strong>${s.name}</strong></td>
      <td class="num">${s.scale}</td>
      <td style="font-size:12px;color:#9b94b0">${s.why}</td>
    </tr>`).join('');

    vp.innerHTML = `<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#4a90c4;margin-bottom:8px">Tier 1 — OPUS-Eligible</div>
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>Dataset</th><th>Scale</th><th>Why</th></tr></thead>
      <tbody>${makeRows(t1)}</tbody>
    </table></div>`;

    dp.innerHTML = `<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#3a9e72;margin-bottom:8px">Tier 2 — Always-On Indic</div>
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>Dataset</th><th>Scale</th><th>Why</th></tr></thead>
      <tbody>${makeRows(t2)}</tbody>
    </table></div>
    <div class="info-box">Custom Indic crawl seeds: <strong>Dainik Jagran, Eenadu, Dinamalar</strong> (news); <strong>*.gov.in, PIB, Lok Sabha transcripts</strong> (govt); <strong>Indian Kanoon, Supreme Court judgments</strong> (legal).</div>`;
  }

  else if (key === 'anneal') {
    vp.innerHTML = `<div class="plot-card" style="width:100%"><h3>Pre-train vs Anneal Mix (% of tokens)</h3>
      ${ANNEAL_DOMAINS.map(d => `
        <div class="anneal-row">
          <div class="anneal-domain">${d.name}</div>
          <div class="anneal-bars">
            <div class="anneal-bar-row">
              <span class="anneal-bar-label">Pre-train</span>
              <div class="abar-track"><div class="abar-fill abar-pre" style="width:${d.pre}%"></div></div>
              <span class="anneal-pct">${d.pre}%</span>
            </div>
            <div class="anneal-bar-row">
              <span class="anneal-bar-label">Anneal</span>
              <div class="abar-track"><div class="abar-fill abar-ann" style="width:${d.ann}%"></div></div>
              <span class="anneal-pct">${d.ann}%</span>
              <span class="anneal-arrow" style="color:${d.ann>d.pre?'#3a9e72':'#c85a47'}">${d.ann>d.pre?'↑':'↓'}</span>
            </div>
          </div>
        </div>`).join('')}
    </div>`;
    dp.innerHTML = `
      <div class="info-box"><strong>Why anneal?</strong> OLMo 2 moved grade-school math from 24% → 67% in anneal with negligible extra compute. We apply the same principle across math, Indic, and India-specific data.</div>
      ${[
        {l:'Anneal budget',v:'500B unique tokens'},
        {l:'Effective with 4× repeat',v:'~2T tokens'},
        {l:'% of training compute',v:'~5% (cooldown only)'},
        {l:'Checkpoints during anneal',v:'Every 10% = 5 checkpoints'},
        {l:'Repetition safety',v:'4× is in safe zone'},
        {l:'Math synthetic',v:'~50M verified pairs'},
      ].map(s=>`<div class="detail-stat"><span class="detail-stat-label">${s.l}</span><span class="detail-stat-val">${s.v}</span></div>`).join('')}
      <div class="info-box" style="margin-top:8px"><strong>4× repetition is safe:</strong> Anneal data is top-28% of already-filtered corpus. Memorisation of gold-standard examples is desirable here, not harmful.</div>`;
  }
}

// ──────────── STAGE 3: Language table ────────────────────────────
function renderLangTable() {
  const tbody = document.getElementById('lang-tbody');
  tbody.innerHTML = LANGS.map(l => {
    const imp = Math.round((1 - l.target / l.curr) * 100);
    const prioCls = l.prio === 1 ? 'prio-1' : l.prio === 2 ? 'prio-2' : 'prio-3';
    const prioLbl = l.prio === 1 ? 'Tier 1 — High' : l.prio === 2 ? 'Tier 2 — Medium' : 'Tier 3 — Protected';
    return `<tr>
      <td><strong>${l.lang}</strong></td>
      <td style="font-size:12px;color:#9b94b0">${l.script}</td>
      <td class="num">${l.speakers}</td>
      <td><span class="prio-badge ${prioCls}">${prioLbl}</span></td>
      <td class="num" style="color:#c85a47">${l.curr.toFixed(1)}</td>
      <td class="num" style="color:#3a9e72"><strong>${l.target.toFixed(1)}</strong></td>
      <td><span class="gap-badge gap-up">-${imp}%</span></td>
      <td class="num">${l.vocab ? l.vocab.toLocaleString() : '(shared)'}</td>
    </tr>`;
  }).join('');
}

// ──────────── STAGE 4: Pipeline ──────────────────────────────────
function renderPipeline(activeIdx) {
  const stepsEl  = document.getElementById('pipeline-steps');
  const detailEl = document.getElementById('pipeline-detail');

  stepsEl.innerHTML = PIPELINE.map((s, i) => `
    <div class="pipe-step ${i === activeIdx ? 'active' : ''}" data-idx="${i}" id="pipe-${i}">
      <div class="pipe-step-num">${String(i+1).padStart(2,'0')}</div>
      <div class="pipe-step-name">${s.name}</div>
      <span class="pipe-step-tag">${s.tag}</span>
    </div>`).join('');

  const s = PIPELINE[activeIdx];
  detailEl.innerHTML = `
    <div class="pd-title">Step ${activeIdx+1}: ${s.name}</div>
    <div class="pd-desc">${s.desc}</div>
    <div><span style="font-size:11px;font-weight:700;color:#9b94b0;text-transform:uppercase;letter-spacing:.5px">Removes</span>
      <div class="pd-tags" style="margin-top:6px">${s.removes.map(r=>`<span class="pd-tag">🚫 ${r}</span>`).join('')}</div>
    </div>
    <div class="pd-stat-row">${s.stats.map(t=>`
      <div class="pd-stat"><div class="pd-stat-val">${t.v}</div><div class="pd-stat-label">${t.l}</div></div>`).join('')}
    </div>`;

  // Re-bind clicks
  stepsEl.querySelectorAll('.pipe-step').forEach(el => {
    el.addEventListener('click', () => renderPipeline(+el.dataset.idx));
  });
}

// ──────────── STAGE 5: Post-training tabs ────────────────────────
function renderPostTab(key) {
  const cont = document.getElementById('post-content');

  if (key === 'sft') {
    const maxVol = Math.max(...SFT_TASKS.map(t => t.vol));
    cont.innerHTML = `
      <div class="sft-grid">
        ${SFT_TASKS.map(t => {
          const pct = Math.round(t.vol / t.total * 100);
          return `<div class="sft-card">
            <div style="font-size:22px">${t.icon}</div>
            <div class="sft-card-title">${t.title}</div>
            <div class="sft-card-vol">${(t.vol/1e6).toFixed(1)}M</div>
            <div class="sft-card-label">pairs · ${pct}% of total</div>
            <div class="sft-bar-track"><div class="sft-bar-fill" style="width:${t.vol/maxVol*100}%"></div></div>
            <div style="font-size:11px;color:#9b94b0;margin-top:4px">${t.note}</div>
          </div>`;
        }).join('')}
      </div>
      <div class="info-box"><strong>Quality rule:</strong> All SFT responses scored by Gemma 4 31B as judge (1–5); retain only ≥4. Instruction diversity enforced via embedding cosine similarity &lt;0.85. Max 5K pairs per template.</div>`;
  }

  else if (key === 'rl') {
    cont.innerHTML = `
      <div style="margin-bottom:14px" class="info-box"><strong>Training order:</strong> SFT → DPO (general helpfulness + Indic style + India-perspective) → RLVR with GRPO (math, code, agentic). Final merge via SLERP or continued training on DPO checkpoint. Algorithm: GRPO with N=8 responses/prompt, β=0.05, LR=1e-6.</div>
      <div class="rl-grid">
        ${RL_TRACKS.map(t => `<div class="rl-track">
          <div class="rl-track-title"><span class="rl-track-icon">${t.icon}</span>${t.title}</div>
          <div style="font-size:12px;color:#9b94b0;margin-bottom:6px">${t.sources}</div>
          <div class="rl-reward-box">${t.reward}</div>
        </div>`).join('')}
      </div>`;
  }

  else if (key === 'eval') {
    cont.innerHTML = `
      <div class="firewall-box">
        <div class="firewall-title">🔒 Evaluation Firewall — 3-Way Separation</div>
        <div class="firewall-tiers">
          <div class="fw-tier">🟦 <strong>Training Pools</strong> — Pre-train → Anneal → SFT → DPO → RL</div>
          <div class="fw-tier">🟨 <strong>Benchmark Train Splits</strong> — Allowed in SFT for format familiarity only</div>
          <div class="fw-tier">🔴 <strong>Golden Proxy (TEST SETS)</strong> — NEVER touch training. n-gram hashed Day 1. Monthly probe audit.</div>
        </div>
      </div>
      <div class="eval-grid">
        ${EVAL_CARDS.map(card => `<div class="eval-card">
          <div class="eval-card-title">${card.title}</div>
          ${card.items.map(item => `<div class="eval-row">
            <span class="eval-bench">${item.bench}</span>
            <span class="eval-target">${item.target}</span>
          </div>`).join('')}
        </div>`).join('')}
      </div>
      <div class="info-box" style="margin-top:14px"><strong>★ Custom India benchmarks</strong> — IIT-JEE Bench (5 years of JEE Advanced), NEET-Bench, IndiaKnowledge-1000, LegalIndia-500, and IndiaCosMo-200 are unique to Bharat-40B and held strictly in the golden proxy.</div>`;
  }
}
