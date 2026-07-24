// app.js – ERA Session 4 Webapp Logic

const D = window.PIPELINE_DATA;

// ── Colour helpers ────────────────────────────────────────────────────────
const lighten = c => c + "22";

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────
function toggleSidebar() {
  const sidebar  = document.getElementById("sidebar");
  const overlay  = document.getElementById("sidebar-overlay");
  const isOpen   = sidebar.classList.contains("open");
  sidebar.classList.toggle("open", !isOpen);
  overlay.classList.toggle("visible", !isOpen);
}

// ── Navigation ────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-widget").forEach(btn => {
  btn.addEventListener("click", () => {
    const p = btn.dataset.panel;
    document.querySelectorAll(".nav-widget").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + p).classList.add("active");
    // Close sidebar on mobile after navigation
    if (window.innerWidth <= 768) toggleSidebar();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

function jumpToPanel(n) {
  document.querySelector(`[data-panel="${n}"]`).click();
}

// ── W1: Pipeline Flow ─────────────────────────────────────────────────────
function buildPipelineFlow() {
  const flow = document.getElementById("pipeline-flow");
  const stages = D.wikipedia.stages;
  let html = "";
  stages.forEach((s, i) => {
    const pct = Math.round(s.output / 100000 * 100);
    html += `
      <div class="pipe-node" onclick="jumpToPanel(4); setTimeout(()=>selectStageTab('en',${i}),200)">
        <div class="pipe-bubble" style="background:${s.color}" data-pct="${pct}">
          <span style="font-size:1.3rem">${s.icon}</span>
        </div>
        <div class="pipe-label">${s.name}</div>
        <div class="pipe-count">${s.output.toLocaleString()}</div>
      </div>`;
    if (i < stages.length - 1) html += `<div class="pipe-arrow">&#8594;</div>`;
  });
  flow.innerHTML = html;
}

function buildFunnel() {
  const wrap = document.getElementById("funnel-bars");
  const stages = D.wikipedia.stages;
  const initial = 100000;
  let html = `<div class="funnel-row">
    <div class="funnel-label">Raw Input</div>
    <div class="funnel-bar-bg"><div class="funnel-bar" style="width:100%;background:#6366f1">100,000</div></div>
    <div class="funnel-val" style="color:#6366f1">100.0%</div></div>`;
  stages.forEach(s => {
    const pct = (s.output / initial * 100).toFixed(1);
    const w = Math.max(2, pct);
    html += `<div class="funnel-row">
      <div class="funnel-label">${s.name}</div>
      <div class="funnel-bar-bg">
        <div class="funnel-bar" style="width:${w}%;background:${s.color}">
          ${pct > 20 ? s.output.toLocaleString() : ""}
        </div>
      </div>
      <div class="funnel-val" style="color:${s.color}">${pct}%</div>
    </div>`;
  });
  wrap.innerHTML = html;
  // Animate bars in
  requestAnimationFrame(() => {
    document.querySelectorAll(".funnel-bar").forEach(b => {
      const w = b.style.width;
      b.style.width = "0";
      setTimeout(() => b.style.width = w, 100);
    });
  });
}

// ── W2: Strategy Cards ────────────────────────────────────────────────────
function buildStrategies() {
  const grid = document.getElementById("strategy-cards");
  grid.innerHTML = D.strategies.map(s => `
    <div class="card">
      <div class="card-icon">${s.icon}</div>
      <div class="card-title" style="color:${s.color}">Stage ${s.id}: ${s.name}</div>
      <div class="card-body">
        <strong style="color:#92400e">Pipeline Gap:</strong> ${s.gap}<br/><br/>
        <strong>Fix:</strong> ${s.fix}<br/><br/>
        <strong>Drops:</strong> ${s.drop}
      </div>
      <span class="card-tag" style="background:${s.color}22;color:${s.color}">Strategy ${s.id}</span>
    </div>
  `).join("");
}

// ── W3: Dataset Cards ─────────────────────────────────────────────────────
function buildDatasets() {
  const container = document.getElementById("dataset-cards");
  const wiki = D.wikipedia.meta;
  const sang = D.sangraha.meta;
  container.innerHTML = `
    <div class="dataset-card active">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:1.5rem">📖</span>
        <div class="dataset-title">Wikipedia (English)</div>
      </div>
      <span class="dataset-badge" style="background:#eef2ff;color:#6366f1">PRIMARY RUN</span>
      <div style="font-size:.82rem;color:var(--muted);margin:8px 0;">${wiki.description}</div>
      <div style="font-size:.78rem;margin-bottom:8px;"><strong>License:</strong> ${wiki.license} &nbsp;|&nbsp; <strong>HF Config:</strong> <span class="inline-code">${wiki.config}</span></div>
      <div class="dataset-stat-row">
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#6366f1">${wiki.initial_docs.toLocaleString()}</div><div class="dataset-stat-l">Input Docs</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#10b981">${wiki.admitted_docs.toLocaleString()}</div><div class="dataset-stat-l">Admitted</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#f59e0b">${(wiki.admitted_tokens/1e6).toFixed(1)}M</div><div class="dataset-stat-l">Tokens</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#06b6d4">${wiki.survival_pct}%</div><div class="dataset-stat-l">Survival</div></div>
      </div>
      <div style="margin-top:12px;font-size:.78rem;color:var(--muted);"><strong>Why chosen:</strong> Known noise profile (disambiguation, cite markup, math symbols). Exercises all 8 pipeline stages. Referenced as Tier 1 D1 source in Session 3.</div>
    </div>
    <div class="dataset-card">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:1.5rem">🇮🇳</span>
        <div class="dataset-title">Sangraha (Hindi + Telugu)</div>
      </div>
      <span class="dataset-badge" style="background:#d1fae5;color:#065f46">INDIC DEMO RUN</span>
      <div style="font-size:.82rem;color:var(--muted);margin:8px 0;">${sang.description}</div>
      <div style="font-size:.78rem;margin-bottom:8px;"><strong>License:</strong> ${sang.license} &nbsp;|&nbsp; <strong>Source:</strong> ai4bharat/sangraha (synthetic fallback)</div>
      <div class="dataset-stat-row">
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#6366f1">${sang.initial_docs.toLocaleString()}</div><div class="dataset-stat-l">Input Docs</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#10b981">${sang.admitted_docs.toLocaleString()}</div><div class="dataset-stat-l">Admitted</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#f59e0b">${sang.initial_tokens.toLocaleString()}</div><div class="dataset-stat-l">Input Tokens</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#ef4444">${sang.survival_pct}%</div><div class="dataset-stat-l">Survival</div></div>
      </div>
      <div style="margin-top:12px;font-size:.78rem;color:var(--muted);"><strong>Why chosen:</strong> Demonstrates ZWNJ/ZWJ preservation, Telugu quality filter failure (Sovereign Indic Exception), AADHAAR/PAN PII patterns, and aggressive dedup on repeated web content. Session 3 Tier 2 Always-ON Channel source.</div>
      <div style="margin-top:8px;font-size:.75rem;padding:8px;background:#fef3c7;border-radius:6px;color:#92400e;">${sang.dedup_note}</div>
    </div>`;
}

// ── W4/W5: Stage-by-Stage Transitions ────────────────────────────────────
function buildStageDetails(runKey, tabsId, detailsId) {
  const stages = D[runKey].stages;
  const tabsEl = document.getElementById(tabsId);
  const detailsEl = document.getElementById(detailsId);

  tabsEl.innerHTML = stages.map((s, i) =>
    `<button class="stage-tab${i===0?' active':''}" data-idx="${i}" onclick="selectStageTab('${runKey}',${i})">${s.icon} S${s.id} ${s.name}</button>`
  ).join("");

  detailsEl.innerHTML = stages.map((s, i) => `
    <div class="stage-detail ${i===0?'active':''}" id="${runKey}-detail-${i}">
      <!-- Left: Stats -->
      <div>
        <div class="detail-box" style="margin-bottom:14px;">
          <h3><span style="font-size:1.3rem">${s.icon}</span> Stage ${s.id}: ${s.name} &mdash; Stats</h3>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            <span class="stat-pill" style="background:#eef2ff;color:#6366f1">&#8594; IN: ${s.input.toLocaleString()}</span>
            <span class="stat-pill" style="background:#d1fae5;color:#065f46">&#10003; OUT: ${s.output.toLocaleString()}</span>
            <span class="stat-pill" style="background:#fee2e2;color:#991b1b">&#x2715; DROPPED: ${s.dropped.toLocaleString()} (${s.drop_pct}%)</span>
            <span class="stat-pill" style="background:#fef3c7;color:#92400e">&#128172; Tokens out: ${(s.output_tok/1e6).toFixed(1)}M</span>
          </div>
          <div style="font-size:.83rem;color:var(--muted);margin-bottom:10px;">${s.drop_summary}</div>
          <div style="font-size:.8rem;"><strong>What was removed:</strong><br/><span style="color:var(--muted)">${s.what_removed}</span></div>
          <div style="margin-top:10px;"><strong style="font-size:.78rem;">Metadata added:</strong><br/>${s.metadata_added.map(m=>`<span class="inline-code">${m}</span>`).join(" ")}</div>
        </div>
        <div class="pipeline-gap-alert">${runKey==='wikipedia'
          ? getPipelineNote(s.id)
          : getIndicNote(s.id)}
        </div>
      </div>
      <!-- Right: Example -->
      <div>
        <div class="detail-box">
          <h3>&#x1F4DD; Before / After Example</h3>
          <div class="example-block">
            <div class="example-label">&#x2B55; BEFORE (raw input)</div>
            <div class="example-text">${escHtml(s.example.before)}</div>
          </div>
          <div class="example-arrow">&#x2193;</div>
          <div class="example-block">
            <div class="example-label">&#x2705; AFTER (stage output)</div>
            <div class="example-text" style="color:${s.example.after.startsWith('[') ? '#dc2626' : '#065f46'}">${escHtml(s.example.after)}</div>
          </div>
          <div class="note-badge">&#x1F4A1; ${s.example.note}</div>
        </div>
      </div>
    </div>`
  ).join("");
}

function selectStageTab(runKey, idx) {
  document.querySelectorAll(`#${runKey==='wikipedia'?'en':'in'}-stage-tabs .stage-tab`).forEach(t => t.classList.remove("active"));
  document.querySelectorAll(`[id^="${runKey}-detail-"]`).forEach(d => d.classList.remove("active"));
  document.querySelector(`#${runKey==='wikipedia'?'en':'in'}-stage-tabs [data-idx="${idx}"]`)?.classList.add("active");
  document.getElementById(`${runKey}-detail-${idx}`)?.classList.add("active");
}

// Pipeline gap notes for English (Wikipedia) run
function getPipelineNote(stageId) {
  const notes = {
    1: "<strong>Pipeline Gap fixed:</strong> Earlier approach naively stripped HTML which kept navigation links and cookie banners as document body content.",
    2: "<strong>Pipeline Gap fixed:</strong> No shared clean_text() normalization function existed in any ingestion script &rarr; 46 garbage vocabulary tokens were baked into the tokenizer.",
    3: "<strong>Pipeline Gap fixed:</strong> Trusted directory naming (verified/asm/) without runtime validation. Nested dict LANG_3_TO_2 caused silent language misrouting.",
    4: "<strong>Pipeline Gap fixed:</strong> English-heavy heuristic proxy classifier systematically undervalued Indic scripts. Manual always-on bypass allowed bad noise in.",
    5: "<strong>Pipeline Gap fixed:</strong> Only local per-source deduplication &mdash; no global dedup. Cross-shard duplicates leaked, wasting compute and raising memorization risk.",
    6: "<strong>Pipeline Gap fixed:</strong> Lacked PII scrubbing for Indic pipelines entirely. Personal identifiers were exposed in training set.",
    7: "<strong>Pipeline Gap fixed:</strong> No active decontamination firewall. An 18.7% benchmark collision rate was discovered only post-training, requiring Band B2 deletion.",
    8: "<strong>Pipeline Gap fixed:</strong> Non-deterministic row counters for shard IDs + words&times;1.3 token estimates &rarr; undercount by up to 10&times; for Indic languages."
  };
  return notes[stageId] || "";
}

// Indic-specific insight notes
function getIndicNote(stageId) {
  const notes = {
    1: "<strong>Indic Note:</strong> Clean Indic prose from Sangraha needs no heavy markup stripping &mdash; demonstrates the pipeline adapts to already-clean sources.",
    2: "<strong>Sovereign Indic Exception:</strong> ZWNJ (U+200C) and ZWJ (U+200D) PRESERVED. These control Devanagari, Malayalam, and Telugu ligature formation. Stripping them would corrupt spellings and break tokenization.",
    3: "<strong>Script Histogram Validation:</strong> fastText detects hi/te correctly (100% match). Runtime validation catches mislabelled files that directory-path trust would pass silently.",
    4: "<strong>Sovereign Indic Exception FAILURE:</strong> Telugu uses fullstop '.' but our punctuation filter checks for danda '&#x0964;'. This demonstrates why Indic-specific thresholds are MANDATORY. English Gopher rules must not be applied directly.",
    5: "<strong>Global Dedup Critical for Indic:</strong> Synthetic data exposed: 14,985 exact duplicates removed. Real Indic web crawls have massive news syndication &mdash; the same article appears across 50+ regional news sites. Global dedup is load-bearing.",
    6: "<strong>India-Specific PII:</strong> AADHAAR (12-digit: XXXX XXXX XXXX) and PAN (ABCDE1234F) patterns active. Indic place names like 'Mysuru' must NOT be masked as personal names by overly aggressive NER.",
    7: "<strong>Indic Decontam:</strong> Indic benchmarks (IndicGLUE, AI4Bharat evals) have different golden proxy sets. Clean Indic prose shows zero contamination with English benchmarks.",
    8: "<strong>Bilingual Manifest:</strong> lang_distribution records hi=67%, te=33% to support mixed-script shard tracking. Critical for Indic corpus accounting."
  };
  return notes[stageId] || "";
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── W6: Charts & Stats ────────────────────────────────────────────────────
function buildCharts() {
  const stages = D.wikipedia.stages;
  const labels = stages.map(s => s.name);
  const colors = stages.map(s => s.color);

  // Donut
  const admittedDocs = 95404, droppedDocs = 100000 - 95404;
  new Chart(document.getElementById("chart-donut"), {
    type: "doughnut",
    data: {
      labels: ["Admitted", "Dropped"],
      datasets: [{ data: [admittedDocs, droppedDocs], backgroundColor: ["#10b981","#ef4444"], borderWidth: 2, borderColor: "#fff" }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()} docs (${(ctx.raw/100000*100).toFixed(1)}%)` } }
      }
    }
  });

  // Bar: drop %
  new Chart(document.getElementById("chart-drops"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Drop %", data: stages.map(s => s.drop_pct), backgroundColor: colors, borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "Drop %" }, beginAtZero: true } }
    }
  });

  // Line: survival
  new Chart(document.getElementById("chart-survival"), {
    type: "line",
    data: {
      labels: ["Raw", ...labels],
      datasets: [{
        label: "Token Survival %",
        data: [100, ...stages.map(s => parseFloat((s.output_tok/96406415*100).toFixed(1)))],
        borderColor: "#6366f1", backgroundColor: "#6366f122",
        fill: true, tension: 0.4, pointRadius: 5, pointBackgroundColor: colors
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "%" }, min: 80, max: 101 } }
    }
  });

  // Bar: processing time
  new Chart(document.getElementById("chart-time"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Time (s)", data: stages.map(s => s.processing_time_s || 0), backgroundColor: colors, borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { title: { display: true, text: "Seconds" }, type: "logarithmic" } }
    }
  });

  // Summary table
  const tbody = document.getElementById("summary-tbody");
  tbody.innerHTML = stages.map((s, i) => `
    <tr style="border-bottom:1px solid #f1f5f9;${i%2===1?'background:#f8fafc':''}">
      <td style="padding:10px 14px;font-weight:600;">${s.icon} ${s.name}</td>
      <td style="padding:10px 14px;text-align:right;">${s.input.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:#065f46;font-weight:600">${s.output.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:${s.dropped>0?'#dc2626':'#64748b'}">${s.dropped.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:${s.drop_pct>1?'#f59e0b':'#64748b'}">${s.drop_pct}%</td>
      <td style="padding:10px 14px;text-align:right;color:var(--muted)">${(s.processing_time_s||0).toFixed(1)}s</td>
    </tr>`
  ).join("");
}

// ── W7: Manifest ──────────────────────────────────────────────────────────
function buildManifest() {
  const m = D.sampleManifest;
  const jsonEl = document.getElementById("manifest-json");

  function colorJson(obj, indent=0) {
    const pad = "  ".repeat(indent);
    let html = "";
    Object.entries(obj).forEach(([k, v], i, arr) => {
      const comma = i < arr.length - 1 ? "," : "";
      const key = `<span class="json-key">"${k}"</span>: `;
      if (typeof v === "string") html += `${pad}${key}<span class="json-str">"${v}"</span>${comma}\n`;
      else if (typeof v === "number") html += `${pad}${key}<span class="json-num">${v}</span>${comma}\n`;
      else if (typeof v === "boolean") html += `${pad}${key}<span class="json-bool">${v}</span>${comma}\n`;
      else if (Array.isArray(v)) html += `${pad}${key}[${v.map(x=>`<span class="json-str">"${x}"</span>`).join(", ")}]${comma}\n`;
      else if (typeof v === "object") html += `${pad}${key}{\n${colorJson(v, indent+1)}${pad}}${comma}\n`;
    });
    return html;
  }

  jsonEl.innerHTML = `<span style="color:#94a3b8">{</span>\n${colorJson(m)}<span style="color:#94a3b8">}</span>`;

  // Run summary
  const w = D.wikipedia.meta;
  document.getElementById("run-summary").innerHTML = `
    <div><strong>Wikipedia Run:</strong> ${w.initial_docs.toLocaleString()} docs &rarr; <strong style="color:#10b981">${w.admitted_docs.toLocaleString()} admitted</strong> (${w.survival_pct}% survival rate)</div>
    <div><strong>Tokens admitted:</strong> ${(w.admitted_tokens/1e6).toFixed(1)}M BPE-aware tokens (not word&times;1.3 estimate)</div>
    <div><strong>Total pipeline time:</strong> ~${Math.round(w.total_time_s/60)} minutes (dedup stage dominates: 704s for MinHash LSH)</div>
    <div><strong>PII redactions:</strong> 2,013 tokens across 1,020 documents</div>
    <div><strong>Benchmark overlaps caught:</strong> 171 documents quarantined by decontam firewall</div>
    <div><strong>License:</strong> CC-BY-SA 4.0 — all shards inherit source license</div>
    <div><strong>Script hash:</strong> Recorded in every manifest — pipeline changes detectable</div>
  `;
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildPipelineFlow();
  buildFunnel();
  buildStrategies();
  buildDatasets();
  buildStageDetails("wikipedia", "en-stage-tabs", "en-stage-details");
  buildStageDetails("sangraha",  "in-stage-tabs",  "in-stage-details");
  buildCharts();
  buildManifest();
});