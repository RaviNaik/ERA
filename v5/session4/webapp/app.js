// app.js – ERA Session 4 Webapp Logic

const D = window.PIPELINE_DATA;

// ── Colour helpers ────────────────────────────────────────────────────────
const lighten = c => c + "22";

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  const isOpen = sidebar.classList.contains("open");
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
  const stages = D.c4_crawl.stages;
  const initialDocs = D.c4_crawl.meta.initial_docs;
  let html = "";
  stages.forEach((s, i) => {
    const pct = Math.round(s.output / initialDocs * 100);
    html += `
      <div class="pipe-node" onclick="jumpToPanel(8); setTimeout(()=>selectStageTab('c4_crawl',${i}),200)">
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
  const stages = D.c4_crawl.stages;
  const initial = D.c4_crawl.meta.initial_docs;
  let html = `<div class="funnel-row">
    <div class="funnel-label">Raw Input</div>
    <div class="funnel-bar-bg"><div class="funnel-bar" style="width:100%;background:#6366f1">${initial.toLocaleString()}</div></div>
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
  const c4   = D.c4_crawl.meta;
  const c4Lang = D.c4_crawl.stages.find(s => s.name === "Language ID");
  const c4Qual = D.c4_crawl.stages.find(s => s.name === "Quality Filter");
  const c4Pii  = D.c4_crawl.stages.find(s => s.name === "PII Scrub");
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
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#f59e0b">${(wiki.admitted_tokens / 1e6).toFixed(1)}M</div><div class="dataset-stat-l">Tokens</div></div>
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
      <div style="font-size:.78rem;margin-bottom:8px;"><strong>License:</strong> ${sang.license} &nbsp;|&nbsp; <strong>Source:</strong> ${sang.source}</div>
      <div class="dataset-stat-row">
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#6366f1">${sang.initial_docs.toLocaleString()}</div><div class="dataset-stat-l">Input Docs</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#10b981">${sang.admitted_docs.toLocaleString()}</div><div class="dataset-stat-l">Admitted</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#f59e0b">${sang.initial_tokens.toLocaleString()}</div><div class="dataset-stat-l">Input Tokens</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#ef4444">${sang.survival_pct}%</div><div class="dataset-stat-l">Survival</div></div>
      </div>
      <div style="margin-top:12px;font-size:.78rem;color:var(--muted);"><strong>Why chosen:</strong> Demonstrates ZWNJ/ZWJ preservation, Telugu quality filter failure (Sovereign Indic Exception), AADHAAR/PAN PII patterns, and aggressive dedup on repeated web content. Session 3 Tier 2 Always-ON Channel source.</div>
      <div style="margin-top:8px;font-size:.75rem;padding:8px;background:#fef3c7;border-radius:6px;color:#92400e;">${sang.dedup_note}</div>
    </div>
    <div class="dataset-card">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:1.5rem">🌐</span>
        <div class="dataset-title">C4 Web Crawl (en.noclean)</div>
      </div>
      <span class="dataset-badge" style="background:#fef2f2;color:#dc2626">REAL WEB CRAWL — ${c4.survival_pct}% SURVIVAL</span>
      <div style="font-size:.82rem;color:var(--muted);margin:8px 0;">${c4.description}</div>
      <div style="font-size:.78rem;margin-bottom:8px;"><strong>License:</strong> ${c4.license} &nbsp;|&nbsp; <strong>HF Config:</strong> <span class="inline-code">allenai/c4 en.noclean</span></div>
      <div class="dataset-stat-row">
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#6366f1">${c4.initial_docs.toLocaleString()}</div><div class="dataset-stat-l">Input Docs</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#10b981">${c4.admitted_docs.toLocaleString()}</div><div class="dataset-stat-l">Admitted</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#f59e0b">${(c4.admitted_tokens / 1e6).toFixed(1)}M</div><div class="dataset-stat-l">Tokens</div></div>
        <div class="dataset-stat"><div class="dataset-stat-v" style="color:#dc2626;font-weight:800">${c4.survival_pct}%</div><div class="dataset-stat-l">Survival</div></div>
      </div>
      <div style="margin-top:8px;font-size:.78rem;padding:10px;background:#fef2f2;border-radius:8px;color:#991b1b;">
        <strong>Stage 3 (Lang ID):</strong> –${c4Lang.drop_pct}% &nbsp;|&nbsp; <strong>Stage 4 (Quality):</strong> –${c4Qual.drop_pct}% &nbsp;|&nbsp; <strong>PII redactions:</strong> ${c4Pii.drop_summary}<br/>
        This is why raw web crawls survive far less than pre-curated sources. Wikipedia's ${D.wikipedia.meta.survival_pct}% is the exception, not the rule.
      </div>
    </div>`;
}

// ── W4/W5: Stage-by-Stage Transitions ────────────────────────────────────
function buildStageDetails(runKey, tabsId, detailsId) {
  const stages = D[runKey].stages;
  const tabsEl = document.getElementById(tabsId);
  const detailsEl = document.getElementById(detailsId);

  tabsEl.innerHTML = stages.map((s, i) =>
    `<button class="stage-tab${i === 0 ? ' active' : ''}" data-idx="${i}" onclick="selectStageTab('${runKey}',${i})">${s.icon} S${s.id} ${s.name}</button>`
  ).join("");

  detailsEl.innerHTML = stages.map((s, i) => `
    <div class="stage-detail ${i === 0 ? 'active' : ''}" id="${runKey}-detail-${i}">
      <!-- Left: Stats -->
      <div>
        <div class="detail-box" style="margin-bottom:14px;">
          <h3><span style="font-size:1.3rem">${s.icon}</span> Stage ${s.id}: ${s.name} &mdash; Stats</h3>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            <span class="stat-pill" style="background:#eef2ff;color:#6366f1">&#8594; IN: ${s.input.toLocaleString()}</span>
            <span class="stat-pill" style="background:#d1fae5;color:#065f46">&#10003; OUT: ${s.output.toLocaleString()}</span>
            <span class="stat-pill" style="background:#fee2e2;color:#991b1b">&#x2715; DROPPED: ${s.dropped.toLocaleString()} (${s.drop_pct}%)</span>
            <span class="stat-pill" style="background:#fef3c7;color:#92400e">&#128172; Tokens out: ${(s.output_tok / 1e6).toFixed(1)}M</span>
          </div>
          <div style="font-size:.83rem;color:var(--muted);margin-bottom:10px;">${s.drop_summary}</div>
          <div style="font-size:.8rem;"><strong>What was removed:</strong><br/><span style="color:var(--muted)">${s.what_removed}</span></div>
          ${s.metadata_added ? `<div style="margin-top:10px;"><strong style="font-size:.78rem;">Metadata added:</strong><br/>${s.metadata_added.map(m => `<span class="inline-code">${m}</span>`).join(" ")}</div>` : ''}
        </div>
        <div class="pipeline-gap-alert">${runKey === 'wikipedia'
      ? getPipelineNote(s.id)
      : runKey === 'sangraha'
        ? getIndicNote(s.id)
        : getC4Note(s.id)}
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
  // Map runKey to the DOM prefix used for tab containers
  const tabPrefix = runKey === 'wikipedia' ? 'en'
                  : runKey === 'sangraha'  ? 'in'
                  : 'c4';
  document.querySelectorAll(`#${tabPrefix}-stage-tabs .stage-tab`).forEach(t => t.classList.remove("active"));
  document.querySelectorAll(`[id^="${runKey}-detail-"]`).forEach(d => d.classList.remove("active"));
  document.querySelector(`#${tabPrefix}-stage-tabs [data-idx="${idx}"]`)?.classList.add("active");
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

// C4 web crawl insight notes -- numbers pulled live from D.c4_crawl so they
// never drift from whatever the pipeline actually produced on its last run.
function getC4Note(stageId) {
  const st = D.c4_crawl.stages[stageId - 1];
  const wikiPii = (D.wikipedia.stages.find(s => s.name === "PII Scrub") || {}).drop_summary || "";
  const notes = {
    1: `<strong>Web Crawl Reality:</strong> ${st.dropped.toLocaleString()} pages were pure navigation menus or stubs — no prose whatsoever after strip. This is common in unfiltered CommonCrawl where many 'pages' are just nav structures baked into one doc.`,
    2: `<strong>Real-world HTML noise:</strong> Real web pages carry far more HTML entities and Unicode noise chars than curated sources like Wikipedia (see counts above).`,
    3: `<strong>${st.drop_pct}% dropped at Lang ID:</strong> Unfiltered CommonCrawl has many non-English pages mixed in. Low-confidence mixed-script nav pages (e.g., English page with foreign-language menu items) also fail the 0.70 confidence threshold.`,
    4: `<strong>${st.drop_pct}% dropped at Quality Filter:</strong> Real nav boilerplate fails avg_line_len (<30 chars). Real SEO spam fails punct_end_ratio (<40%). Real forum/link pages fail multiple rules simultaneously. This is the major kill-zone in web crawl cleaning.`,
    5: `<strong>Dedup catches templated content:</strong> XML sitemap clones and templated product manual pages (identical boilerplate, only serial number differs) caught by both exact SHA-256 and MinHash LSH. Real web crawls have much higher near-dup rates than curated sources.`,
    6: `<strong>PII in the wild:</strong> Real web pages include business contact pages, forum posts with phone numbers, e-commerce pages with customer emails — this is why PII scrubbing is load-bearing for web-scale data.`,
    7: `<strong>Benchmark contamination in real data:</strong> ${st.dropped.toLocaleString()} real web pages contained MCQ patterns (e.g., (a)..(b)..(c)..(d) option lists in quiz pages) and 'how many...total' patterns in news articles. Decontamination firewall correctly quarantined them.`,
    8: `<strong>Final yield: ${D.c4_crawl.meta.survival_pct}% (${D.c4_crawl.meta.admitted_docs.toLocaleString()}/${D.c4_crawl.meta.initial_docs.toLocaleString()}):</strong> This is the trainer's benchmark number made concrete with real data. Stage 3 and Stage 4 do the heavy lifting. The remaining stages refine without major drops.`
  };
  return notes[stageId] || "";
}


// Indic-specific insight notes -- numbers pulled live from D.sangraha
function getIndicNote(stageId) {
  const st = D.sangraha.stages[stageId - 1];
  const langStage = D.sangraha.stages[2]; // stage 3
  const notes = {
    1: "<strong>Indic Note:</strong> Real Sangraha web + PDF-extracted text needs markup stripping just like any other web source &mdash; PDF-extraction artifacts in particular need cleanup.",
    2: "<strong>Sovereign Indic Exception:</strong> ZWNJ (U+200C) and ZWJ (U+200D) PRESERVED. These control Devanagari, Malayalam, and Telugu ligature formation. Stripping them would corrupt spellings and break tokenization.",
    3: `<strong>Script Histogram Validation:</strong> fastText + Devanagari/Telugu codepoint-block validation run on every document (${langStage ? langStage.drop_pct : '?'}% quarantined). Runtime validation catches mislabelled files that directory-path trust would pass silently.`,
    4: "<strong>Sovereign Indic Exception:</strong> Telugu and Hindi word-length upper bound is relaxed to 15 chars (vs English's 10) due to agglutinative morphology, and the punctuation-end check accepts both '.' and the Devanagari danda '।'. English Gopher thresholds applied directly would incorrectly penalize high-quality Indic documents.",
    5: "<strong>Global Dedup Critical for Indic:</strong> Real Indic web crawls have massive news syndication &mdash; the same article appears across many regional news sites. Global (not per-source) dedup is load-bearing for catching this.",
    6: "<strong>India-Specific PII:</strong> AADHAAR (12-digit: XXXX XXXX XXXX) and PAN (ABCDE1234F) patterns active. Indic place names like 'Mysuru' must NOT be masked as personal names by overly aggressive NER.",
    7: "<strong>Indic Decontam:</strong> Indic benchmarks (IndicGLUE, AI4Bharat evals) have different golden proxy sets from MMLU/GSM8K/HumanEval, so the same n-gram/MCQ firewall is applied without special-casing script.",
    8: "<strong>Bilingual Manifest:</strong> lang_distribution records the hi/te split per shard to support mixed-script corpus accounting."
  };
  return notes[stageId] || "";
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── W6: Charts & Stats ────────────────────────────────────────────────────
function buildCharts() {
  const stages = D.wikipedia.stages;
  const labels = stages.map(s => s.name);
  const colors = stages.map(s => s.color);
  const initialDocs = D.wikipedia.meta.initial_docs;
  const initialTok = D.wikipedia.meta.initial_tokens;

  // Donut
  const admittedDocs = D.wikipedia.meta.admitted_docs, droppedDocs = initialDocs - admittedDocs;
  new Chart(document.getElementById("chart-donut"), {
    type: "doughnut",
    data: {
      labels: ["Admitted", "Dropped"],
      datasets: [{ data: [admittedDocs, droppedDocs], backgroundColor: ["#10b981", "#ef4444"], borderWidth: 2, borderColor: "#fff" }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()} docs (${(ctx.raw / initialDocs * 100).toFixed(1)}%)` } }
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
        data: [100, ...stages.map(s => parseFloat((s.output_tok / initialTok * 100).toFixed(1)))],
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
    <tr style="border-bottom:1px solid #f1f5f9;${i % 2 === 1 ? 'background:#f8fafc' : ''}">
      <td style="padding:10px 14px;font-weight:600;">${s.icon} ${s.name}</td>
      <td style="padding:10px 14px;text-align:right;">${s.input.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:#065f46;font-weight:600">${s.output.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:${s.dropped > 0 ? '#dc2626' : '#64748b'}">${s.dropped.toLocaleString()}</td>
      <td style="padding:10px 14px;text-align:right;color:${s.drop_pct > 1 ? '#f59e0b' : '#64748b'}">${s.drop_pct}%</td>
      <td style="padding:10px 14px;text-align:right;color:var(--muted)">${(s.processing_time_s || 0).toFixed(1)}s</td>
    </tr>`
  ).join("");
}

// ── W7: Manifest ──────────────────────────────────────────────────────────
function buildManifest() {
  const m = D.sampleManifest;
  const jsonEl = document.getElementById("manifest-json");

  function colorJson(obj, indent = 0) {
    const pad = "  ".repeat(indent);
    let html = "";
    Object.entries(obj).forEach(([k, v], i, arr) => {
      const comma = i < arr.length - 1 ? "," : "";
      const key = `<span class="json-key">"${k}"</span>: `;
      if (typeof v === "string") html += `${pad}${key}<span class="json-str">"${v}"</span>${comma}\n`;
      else if (typeof v === "number") html += `${pad}${key}<span class="json-num">${v}</span>${comma}\n`;
      else if (typeof v === "boolean") html += `${pad}${key}<span class="json-bool">${v}</span>${comma}\n`;
      else if (Array.isArray(v)) html += `${pad}${key}[${v.map(x => `<span class="json-str">"${x}"</span>`).join(", ")}]${comma}\n`;
      else if (typeof v === "object") html += `${pad}${key}{\n${colorJson(v, indent + 1)}${pad}}${comma}\n`;
    });
    return html;
  }

  jsonEl.innerHTML = `<span style="color:#94a3b8">{</span>\n${colorJson(m)}<span style="color:#94a3b8">}</span>`;

  // Run summary
  const w = D.wikipedia.meta;
  const stages = D.wikipedia.stages;
  const piiStage = stages.find(s => s.name === "PII Scrub");
  const decontamStage = stages.find(s => s.name === "Decontaminate");
  const dedupStage = stages.find(s => s.name === "Deduplicate");
  const piiCount = (piiStage && piiStage.drop_summary.match(/[\d,]+/)) ? piiStage.drop_summary.match(/[\d,]+/)[0] : "n/a";
  document.getElementById("run-summary").innerHTML = `
    <div><strong>Wikipedia Run:</strong> ${w.initial_docs.toLocaleString()} docs &rarr; <strong style="color:#10b981">${w.admitted_docs.toLocaleString()} admitted</strong> (${w.survival_pct}% survival rate)</div>
    <div><strong>Tokens admitted:</strong> ${(w.admitted_tokens / 1e6).toFixed(1)}M tokens (word-based estimate, consistent across all stages)</div>
    <div><strong>Total pipeline time:</strong> ~${Math.round(w.total_time_s / 60)} minutes (dedup stage: ${dedupStage ? dedupStage.processing_time_s.toFixed(0) : '?'}s for MinHash LSH)</div>
    <div><strong>PII redactions:</strong> ${piiStage ? piiStage.drop_summary : 'n/a'}</div>
    <div><strong>Benchmark overlaps caught:</strong> ${decontamStage ? decontamStage.dropped.toLocaleString() : '?'} documents quarantined by decontam firewall</div>
    <div><strong>License:</strong> ${w.license} — all shards inherit source license</div>
    <div><strong>Script hash:</strong> Recorded in every manifest — pipeline changes detectable</div>
  `;
}

// ── Header / Sidebar dynamic text (all derived from D, never hardcoded) ──
function populateHeaderStats() {
  const w = D.wikipedia.meta, s = D.sangraha.meta, c = D.c4_crawl.meta;
  const piiStage = D.wikipedia.stages.find(st => st.name === "PII Scrub");
  const piiCount = piiStage ? (piiStage.drop_summary.match(/[\d,]+/) || ["n/a"])[0] : "n/a";

  document.getElementById("kpi-docs").textContent = w.admitted_docs.toLocaleString();
  document.getElementById("kpi-tokens").textContent = (w.admitted_tokens / 1e6).toFixed(1) + "M";
  document.getElementById("kpi-survival").textContent = w.survival_pct + "%";
  document.getElementById("kpi-pii").textContent = piiCount;

  document.getElementById("nav-c4-desc").textContent = `C4 en.noclean · ${c.survival_pct}% survival`;
  document.getElementById("sidebar-footer-text").textContent =
    `Wikipedia ${(w.initial_docs/1000).toFixed(0)}K · Indic ${(s.initial_docs/1000).toFixed(0)}K · Web ${(c.initial_docs/1000).toFixed(0)}K`;
  document.getElementById("funnel-title").textContent =
    `▼ Survival Funnel — Web Crawl Run (C4 en.noclean, ${c.initial_docs.toLocaleString()} pages)`;
  document.getElementById("wiki-panel-summary").textContent =
    `${w.initial_docs.toLocaleString()} English Wikipedia articles processed through all 8 stages. Click each stage to see what was removed, why, and real before/after examples.`;

  const c4Lang = D.c4_crawl.stages.find(st => st.name === "Language ID");
  const c4Qual = D.c4_crawl.stages.find(st => st.name === "Quality Filter");
  document.getElementById("c4-why-text").innerHTML =
    `Stage 3 (Lang ID) drops ${c4Lang ? c4Lang.drop_pct : '?'}% and Stage 4 (Quality) drops ${c4Qual ? c4Qual.drop_pct : '?'}%, yielding only ${c.survival_pct}% survival.`;

  document.getElementById("c4-panel-summary").innerHTML =
    `${c.initial_docs.toLocaleString()} unfiltered CommonCrawl pages run through the identical 8-stage pipeline. Stage 3 (Lang ID) dropped <strong>${c4Lang ? c4Lang.drop_pct : '?'}%</strong> of docs due to multilingual noise and mixed-script nav pages. Stage 4 (Quality Filter) dropped <strong>${c4Qual ? c4Qual.drop_pct : '?'}%</strong> due to nav boilerplate, SEO spam and forum link lists. Final survival: <strong class="highlight-red">${c.survival_pct}%</strong>.`;
  document.getElementById("c4-panel-alert").innerHTML =
    `&#x1F4A1; <strong>Key insight:</strong> Wikipedia survives at ${w.survival_pct}% because it is a <em>pre-curated source</em>. Unfiltered CommonCrawl (en.noclean) survives at only ${c.survival_pct}% — the pipeline is doing real work.`;
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  populateHeaderStats();
  buildPipelineFlow();
  buildFunnel();
  buildStrategies();
  buildDatasets();
  buildStageDetails("wikipedia", "en-stage-tabs", "en-stage-details");
  buildStageDetails("sangraha", "in-stage-tabs", "in-stage-details");
  buildStageDetails("c4_crawl", "c4-stage-tabs", "c4-stage-details");
  buildCharts();
  buildManifest();
});