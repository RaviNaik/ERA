/* ============================================================
   app.js — renders the hero stats, thread legend, chronological
   timeline, cheat sheet and reference list from data.js, and
   wires up every interactive widget (flagship + embedded
   mini-widgets), plus the narrative scaffolding: era interludes,
   scroll-reveal, reading progress, and the hero canvas.
   ============================================================ */

(function () {
  "use strict";

  const DATA = window.SESSION_DATA;
  const { THREADS, TIMELINE, SOURCES, REF_CONFIG } = DATA;
  const threadById = Object.fromEntries(THREADS.map(t => [t.id, t]));

  /* ── small utilities ─────────────────────────────────────── */
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmaxRow(row) {
    const finite = row.filter(v => Number.isFinite(v));
    const max = finite.length ? Math.max(...finite) : 0;
    const exps = row.map(v => Number.isFinite(v) ? Math.exp(v - max) : 0);
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map(v => v / sum);
  }
  function fmtInt(n) { return Math.round(n).toLocaleString("en-US"); }
  function fmtCompact(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(Math.round(n));
  }
  function fmtBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + " MB";
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + " KB";
    return bytes.toFixed(0) + " B";
  }
  function el(id) { return document.getElementById(id); }
  function esc(s) { return (s ?? "").toString(); }

  /* ── Hero stats ──────────────────────────────────────────── */
  function renderHeroStats() {
    const total = TIMELINE.length;
    const communitySourced = TIMELINE.filter(n => n.isCommunitySource).length;
    el("stat-count").textContent = total;
    el("stat-range").textContent = "2017 → 2025";
    el("stat-sourced").textContent = total + " / " + total;
    el("stat-flagged").textContent = communitySourced;
    el("stat-threads").textContent = THREADS.length;
  }

  /* ── Thread legend ───────────────────────────────────────── */
  let activeThreads = new Set();

  function renderStaticLegend(containerId) {
    const c = el(containerId);
    if (!c) return;
    c.innerHTML = THREADS.map(t =>
      `<span class="thread-pill" style="cursor:default;">
         <span class="dot" style="background:${t.color}"></span>${t.label}
       </span>`
    ).join("");
  }

  function renderInteractiveLegend(containerId, onChange) {
    const c = el(containerId);
    if (!c) return;
    function draw() {
      c.innerHTML = THREADS.map(t => {
        const isActive = activeThreads.has(t.id);
        return `<span class="thread-pill ${isActive ? "active" : ""}" data-thread="${t.id}" style="${isActive ? `color:${t.color}` : ""}">
                  <span class="dot" style="background:${t.color}"></span>${t.label}
                </span>`;
      }).join("") + `<span class="thread-pill thread-pill-reset" data-reset="1">↺ Show all</span>`;
      c.querySelectorAll("[data-thread]").forEach(pillEl => {
        pillEl.addEventListener("click", () => {
          const id = pillEl.dataset.thread;
          if (activeThreads.has(id)) activeThreads.delete(id); else activeThreads.add(id);
          draw();
          onChange();
        });
      });
      const resetEl = c.querySelector("[data-reset]");
      if (resetEl) resetEl.addEventListener("click", () => { activeThreads.clear(); draw(); onChange(); });
    }
    draw();
  }

  /* ── Timeline rendering ─────────────────────────────────── */
  function threadTagsHTML(threadIds) {
    return threadIds.map(id => {
      const t = threadById[id];
      if (!t) return "";
      return `<span class="tl-thread-tag" style="background:${t.color}22;color:${t.color};border:1px solid ${t.color}55;">${t.label}</span>`;
    }).join("");
  }

  function sourceLineHTML(node) {
    const s = SOURCES[node.sourceId];
    if (!s) return "";
    const linkOpen = s.url ? `<a href="${s.url}" target="_blank" rel="noopener">${s.id}</a>` : `<span class="mono">${s.id}</span>`;
    let extra = "";
    if (node.secondarySourceIds && node.secondarySourceIds.length) {
      extra = " · also: " + node.secondarySourceIds.map(id => {
        const ss = SOURCES[id];
        return ss ? (ss.url ? `<a href="${ss.url}" target="_blank" rel="noopener">${ss.id}</a>` : ss.id) : "";
      }).join(", ");
    }
    return `<span class="tl-source">${esc(s.authors)} — “${esc(s.title)}” (${esc(s.venue)}) · ${linkOpen}${extra}</span>`;
  }

  function fixedByHTML(node) {
    if (!node.fixedBy || !node.fixedBy.length) return "";
    const titles = node.fixedBy.map(id => {
      const t = TIMELINE.find(n => n.id === id);
      return t ? t.title : id;
    });
    return `<span class="tl-fixedby">Addressed later by: <span>${titles.join(", ")}</span></span>`;
  }

  function eraInterludeHTML(node) {
    if (!node.era) return "";
    return `<div class="era-interlude reveal">
      <div class="era-title">${esc(node.era.title)}</div>
      <div class="era-blurb">${esc(node.era.blurb)}</div>
    </div>`;
  }

  function cardHTML(node, index) {
    const badges = [];
    if (node.isBaseline) badges.push(`<span class="badge badge-blue">Where it all starts</span>`);
    if (node.isBonus) badges.push(`<span class="badge badge-purple">Not on the required list — included anyway</span>`);
    if (node.isCommunitySource) badges.push(`<span class="badge badge-yellow">Community post, not peer-reviewed</span>`);

    const nodeClasses = ["tl-node", "reveal"];
    if (node.isBaseline) nodeClasses.push("is-baseline");

    const dotColor = threadById[node.threads[0]] ? threadById[node.threads[0]].color : "var(--indigo)";

    return `${eraInterludeHTML(node)}
      <div class="${nodeClasses.join(" ")}" id="tl-${node.id}" data-threads="${node.threads.join(",")}">
        <div class="tl-dot" style="border-color:${dotColor}"></div>
        <div class="tl-card">
          <div class="tl-head">
            <div class="tl-headrow">
              <span class="tl-date">${esc(node.dateDisplay)}</span>
              <div class="tl-threads">${threadTagsHTML(node.threads)}</div>
              ${badges.join(" ")}
            </div>
            <div class="tl-title">${index}. ${esc(node.title)}</div>
            <div class="tl-tagline">${esc(node.tagline)}</div>
          </div>
          <div class="tl-body">
            <div class="tl-block is-lede">
              <span class="tl-block-label">The situation</span>
              <span class="tl-block-text">${esc(node.problem)}</span>
            </div>
            <div class="tl-block">
              <span class="tl-block-label">The idea</span>
              <span class="tl-block-text">${esc(node.mechanism)}</span>
            </div>
            <div class="tl-triad">
              <div class="tl-triad-item buy"><div class="tl-triad-label">✅ Buys</div><div class="tl-triad-text">${esc(node.buys)}</div></div>
              <div class="tl-triad-item cost"><div class="tl-triad-label">⚠️ Gives up</div><div class="tl-triad-text">${esc(node.costs)}</div></div>
              <div class="tl-triad-item choose"><div class="tl-triad-label">🎯 Choose when</div><div class="tl-triad-text">${esc(node.chooseWhen)}</div></div>
            </div>
            ${node.footnote ? `<div class="tl-footnote"><strong>Note —</strong> ${esc(node.footnote)}</div>` : ""}
            <div id="mini-widget-${node.id}"></div>
            <div class="tl-foot">
              ${sourceLineHTML(node)}
              ${fixedByHTML(node)}
            </div>
          </div>
        </div>
      </div>`;
  }

  function renderTimeline() {
    const container = el("timeline-container");
    if (!container) return;
    let i = 0;
    container.innerHTML = TIMELINE.map(node => {
      i++;
      return cardHTML(node, i);
    }).join("");

    // Mount embedded mini-widgets now that DOM exists.
    initRopeWidget("rope");
    initDeltaWidget("delta-rule");
    initSinksWidget("attention-sinks");
    initGqaWidget("gqa");
    initMlaWidget("mla");
    initNsaWidget("deepseek-nsa");
  }

  function applyTimelineFilter() {
    document.querySelectorAll(".tl-node").forEach(nodeEl => {
      if (activeThreads.size === 0) { nodeEl.classList.remove("dimmed"); return; }
      const threads = (nodeEl.dataset.threads || "").split(",");
      const match = threads.some(t => activeThreads.has(t));
      nodeEl.classList.toggle("dimmed", !match);
    });
  }

  /* ── Cheat sheet ─────────────────────────────────────────── */
  function renderCheatSheet() {
    const tbody = el("cheat-sheet-tbody");
    if (!tbody) return;
    tbody.innerHTML = TIMELINE.map(node => {
      const tags = node.threads.map(id => {
        const t = threadById[id];
        return `<span class="cheat-tag" style="background:${t.color}22;color:${t.color};">${t.label}</span>`;
      }).join("");
      return `<tr>
        <td>${esc(node.title)}<br/>${tags}</td>
        <td class="mono">${esc(node.dateDisplay)}</td>
        <td>${esc(node.buys)}</td>
        <td>${esc(node.costs)}</td>
        <td>${esc(node.chooseWhen)}</td>
      </tr>`;
    }).join("");
  }

  /* ── Reference list ──────────────────────────────────────── */
  function renderReferenceList() {
    const list = el("reference-list");
    if (!list) return;
    list.innerHTML = TIMELINE.map(node => {
      const s = SOURCES[node.sourceId];
      if (!s) return "";
      const link = s.url ? `<a href="${s.url}" target="_blank" rel="noopener">${s.id}</a>` : `<em>${s.id}</em>`;
      return `<div class="ref-item">
        <strong>${esc(node.dateDisplay)}</strong> — ${esc(s.authors)}, “${esc(s.title)},” ${esc(s.venue)}. ${link}
      </div>`;
    }).join("");
  }

  /* ============================================================
     WIDGET A — Core Mechanism (flagship)
     ============================================================ */
  function initCoreMechanismWidget() {
    const root = el("core-mechanism-widget");
    if (!root) return;

    const TOKENS = ["The", "cat", "sat", "down"];
    const Q = [[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 2], [0, 0, 2, 1]];
    const K = [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 2]];
    const V = [[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]];
    const dk = 4, scaleDenom = Math.sqrt(dk);
    const STEPS = [
      { n: 1, label: "Project" },
      { n: 2, label: "Q · K" },
      { n: 3, label: "Scale" },
      { n: 4, label: "Mask" },
      { n: 5, label: "Softmax" },
      { n: 6, label: "Weighted Sum" },
    ];

    const state = { step: 1, maskOn: true };

    function computeAll() {
      const raw = Q.map(q => K.map(k => dot(q, k)));
      const scaled = raw.map(row => row.map(v => v / scaleDenom));
      const masked = scaled.map((row, i) => row.map((v, j) => (state.maskOn && j > i) ? -Infinity : v));
      const weights = masked.map(row => softmaxRow(row));
      const output = weights.map(w => {
        const out = [0, 0, 0, 0];
        w.forEach((wi, j) => { for (let d = 0; d < 4; d++) out[d] += wi * V[j][d]; });
        return out;
      });
      return { raw, scaled, masked, weights, output };
    }

    function vecHTML(vec, decimals) {
      return `<div class="cm-vec">${vec.map(v => `<span>${decimals !== undefined ? v.toFixed(decimals) : v}</span>`).join("")}</div>`;
    }

    function tokensPanelHTML() {
      return `<div class="cm-tokens">${TOKENS.map((tok, i) => `
        <div>
          <div class="cm-token-row"><span class="cm-token-label">${tok}</span>${vecHTML(Q[i])}<span class="mono" style="font-size:9px;color:var(--text-muted);align-self:center;">Q</span></div>
          <div class="cm-token-row"><span class="cm-token-label"></span>${vecHTML(K[i])}<span class="mono" style="font-size:9px;color:var(--text-muted);align-self:center;">K</span></div>
          <div class="cm-token-row"><span class="cm-token-label"></span>${vecHTML(V[i])}<span class="mono" style="font-size:9px;color:var(--text-muted);align-self:center;">V</span></div>
        </div>`).join("<div style=\"height:6px;\"></div>")}</div>`;
    }

    function matrixHTML(matrix, opts) {
      opts = opts || {};
      const axis = `<div class="cm-axis"><span></span>${TOKENS.map(t => `<span>${t}</span>`).join("")}</div>`;
      const rows = matrix.map((row, i) => {
        const cells = row.map((v, j) => {
          const isFuture = j > i;
          const masked = opts.showMask && state.maskOn && isFuture;
          const leak = opts.showMask && !state.maskOn && isFuture;
          const txt = masked ? "−∞" : (Number.isFinite(v) ? v.toFixed(2) : "−∞");
          let bg = "";
          if (!masked && Number.isFinite(v) && opts.heat) {
            const t = Math.max(0, Math.min(1, (v - opts.heat.min) / (opts.heat.max - opts.heat.min || 1)));
            bg = `background:rgba(99,102,241,${(0.08 + t * 0.35).toFixed(2)});`;
          }
          return `<div class="cm-cell ${masked ? "masked" : ""} ${leak ? "leak" : ""}" style="${bg}">${txt}</div>`;
        }).join("");
        return `<div style="display:flex; align-items:center; gap:6px;">
                  <span class="cm-axis" style="margin:0;"><span style="min-width:40px;font-weight:700;color:var(--text-primary);">${TOKENS[i]}</span></span>
                  <div class="cm-matrix-row" style="grid-template-columns:repeat(${row.length},1fr);flex:1;">${cells}</div>
                </div>`;
      }).join("<div style='height:4px;'></div>");
      return `<div class="cm-matrix"><div class="cm-axis" style="padding-left:46px;">${TOKENS.map(t => `<span>${t}</span>`).join("")}</div>${rows}</div>`;
    }

    function render() {
      const { raw, scaled, masked, weights, output } = computeAll();

      const stepsHTML = STEPS.map(s => `
        <button class="cm-step-btn ${state.step === s.n ? "active" : ""} ${state.step > s.n ? "done" : ""}" data-step="${s.n}">
          <span class="n">${s.n}</span>${s.label}
        </button>`).join("");

      let rightHTML = "";
      let explainHTML = "";

      if (state.step === 1) {
        rightHTML = `<div class="formula-box">x ──┬── Wq ──▶ query
    ├── Wk ──▶ key
    └── Wv ──▶ value<span class="fx-comment">  (three learned projections of the same vector)</span></div>`;
        explainHTML = `Every token becomes <strong>three</strong> different vectors — a query (“what am I looking for?”), a key (“what do I contain?”) and a value (“what do I hand over if chosen?”). The left panel shows the actual Q/K/V vectors this widget uses for four tokens, computed for a tiny 4-dimensional head.`;
      } else if (state.step === 2) {
        rightHTML = matrixHTML(raw, { heat: { min: Math.min(...raw.flat()), max: Math.max(...raw.flat()) } });
        explainHTML = `<strong>Score = Q · Kᵀ.</strong> Every query is compared against every key with a dot product — one number per pair. Rows are queries (the token asking), columns are keys (the token being asked about). Brighter cells = higher raw score.`;
      } else if (state.step === 3) {
        rightHTML = matrixHTML(scaled, { heat: { min: Math.min(...scaled.flat()), max: Math.max(...scaled.flat()) } });
        explainHTML = `<strong>Divide every score by √d_k = ${scaleDenom.toFixed(0)}.</strong> As the query/key width grows, raw dot products tend to grow with it — scaling keeps the numbers in a range softmax handles well instead of saturating.`;
      } else if (state.step === 4) {
        rightHTML = matrixHTML(scaled, { showMask: true });
        explainHTML = state.maskOn
          ? `<strong>Causal mask on.</strong> Every cell where the key's position is <em>after</em> the query's position is forced to −∞ before softmax (hatched cells). softmax(−∞) = 0, so the future receives exactly zero weight.`
          : `<strong>Causal mask off — watch the rose-outlined cells.</strong> Those are scores for <em>future</em> tokens the query should never see. With no mask, softmax will hand them real, non-zero weight in the next step. That leak is exactly the bug the mask exists to prevent.`;
      } else if (state.step === 5) {
        rightHTML = `<div class="cm-out">${weights.map((w, i) => `
          <div>
            <div style="font-size:10.5px;color:var(--text-muted);margin-bottom:3px;">query = <strong style="color:var(--text-primary);">${TOKENS[i]}</strong></div>
            ${w.map((wt, j) => `
              <div class="cm-out-row">
                <span class="mono" style="width:34px;font-size:10px;color:var(--text-muted);">${TOKENS[j]}</span>
                <div class="cm-bar-track"><div class="cm-bar-fill" style="width:${(wt * 100).toFixed(1)}%;"></div></div>
                <span class="mono" style="width:42px;font-size:10px;text-align:right;">${(wt * 100).toFixed(1)}%</span>
              </div>`).join("")}
          </div>`).join("<div style='height:10px;'></div>")}</div>`;
        explainHTML = `<strong>Softmax turns each row of scores into weights that are positive and sum to 100%.</strong> ${state.maskOn ? "Masked (future) positions always land at exactly 0%." : "Notice future positions now hold real, non-zero weight — this is the leak from Step 4, made concrete."}`;
      } else if (state.step === 6) {
        rightHTML = `<div class="cm-tokens">${output.map((o, i) => `
          <div class="cm-token-row"><span class="cm-token-label">${TOKENS[i]}</span>${vecHTML(o, 2)}<span class="mono" style="font-size:9px;color:var(--text-muted);align-self:center;">out</span></div>
        `).join("")}</div>`;
        explainHTML = `<strong>Weighted sum of V.</strong> Each output vector is the softmax weights from Step 5 applied to the value vectors from Step 1 and added together — one new, context-aware vector per token. This is the layer's entire output.`;
      }

      root.innerHTML = `
        <div class="cm-steps">${stepsHTML}</div>
        <div class="cm-layout">
          <div>
            <div style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Q / K / V vectors (constant reference)</div>
            ${tokensPanelHTML()}
          </div>
          <div>
            ${rightHTML}
            <div class="cm-explain">${explainHTML}</div>
            <div class="cm-toggle-row">
              <div class="cm-switch ${state.maskOn ? "on" : ""}" id="cm-mask-switch"></div>
              <span class="cm-toggle-label">Causal mask ${state.maskOn ? "ON — future tokens blocked" : "OFF — try Step 4/5 now"}</span>
            </div>
          </div>
        </div>`;

      root.querySelectorAll(".cm-step-btn").forEach(b => {
        b.addEventListener("click", () => { state.step = parseInt(b.dataset.step, 10); render(); });
      });
      root.querySelector("#cm-mask-switch").addEventListener("click", () => { state.maskOn = !state.maskOn; render(); });
    }

    render();
  }

  /* ============================================================
     WIDGET B — What It Costs (compute vs KV-cache growth)
     ============================================================ */
  function initTwoBillsWidget() {
    const root = el("two-bills-widget");
    if (!root) return;

    const T_MIN = 128, T_MAX = 2_000_000;
    const state = { slider: 55, users: 8 };

    function tFromSlider(s) {
      return Math.round(T_MIN * Math.pow(T_MAX / T_MIN, s / 100));
    }
    function kvCacheBytes(T) {
      return 2 * REF_CONFIG.layers * REF_CONFIG.kvHeads * REF_CONFIG.headDim * T * REF_CONFIG.bytesPerNumber;
    }

    root.innerHTML = `
      <div class="bills-canvas-wrap"><canvas id="bills-canvas" height="220"></canvas></div>
      <div class="bills-legend">
        <span class="bills-legend-item"><span class="bills-legend-dot" style="background:var(--cyan);"></span>Compute (≈ T², shape only)</span>
        <span class="bills-legend-item"><span class="bills-legend-dot" style="background:var(--amber);"></span>KV cache (≈ T, shape only)</span>
      </div>
      <div class="mw-controls" style="margin-top:16px;">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">Context length T <b id="bills-t-val"></b></span>
          <input type="range" id="bills-t-slider" min="0" max="100" value="${state.slider}" />
        </div>
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">Concurrent conversations <b id="bills-u-val"></b></span>
          <input type="range" id="bills-u-slider" min="1" max="32" value="${state.users}" />
        </div>
      </div>
      <div class="bills-readouts">
        <div class="bills-readout"><div class="br-label">Pairwise Q·K scores at T</div><div class="br-val cyan" id="bills-scores"></div></div>
        <div class="bills-readout"><div class="br-label">KV cache · one conversation</div><div class="br-val amber" id="bills-cache-one"></div></div>
        <div class="bills-readout"><div class="br-label">KV cache · all concurrent</div><div class="br-val amber" id="bills-cache-all"></div></div>
      </div>
      <p class="mw-readout" style="margin-top:12px;">Plug in a fairly ordinary mid-size model (48 layers · 8 KV heads · head_dim 128 · bf16) and the arithmetic is blunt: at T=32,768 tokens, one conversation's cache alone already costs <b>6.44 GB</b>.</p>
    `;

    const canvas = el("bills-canvas");
    const ctx = canvas.getContext("2d");

    function draw() {
      const T = tFromSlider(state.slider);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 600, h = 220;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const pad = { l: 40, r: 16, t: 14, b: 26 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
      const isDark = document.documentElement.getAttribute("data-theme") !== "light";
      const gridColor = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.08)";
      const textColor = isDark ? "#64748b" : "#94a3b8";

      // gridlines
      ctx.strokeStyle = gridColor; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.t + (plotH * i) / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      }

      const logMin = Math.log10(T_MIN), logMax = Math.log10(T_MAX);
      function xForT(t) { return pad.l + ((Math.log10(t) - logMin) / (logMax - logMin)) * plotW; }

      const N = 160;
      const computeVals = [], cacheVals = [];
      for (let i = 0; i <= N; i++) {
        const t = T_MIN * Math.pow(T_MAX / T_MIN, i / N);
        computeVals.push(t * t);
        cacheVals.push(kvCacheBytes(t));
      }
      const cMin = Math.log10(Math.min(...computeVals)), cMax = Math.log10(Math.max(...computeVals));
      const kMin = Math.log10(Math.min(...cacheVals)), kMax = Math.log10(Math.max(...cacheVals));

      function plotLine(vals, minL, maxL, color) {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
          const t = T_MIN * Math.pow(T_MAX / T_MIN, i / N);
          const x = xForT(t);
          const norm = (Math.log10(vals[i]) - minL) / (maxL - minL || 1);
          const y = pad.t + plotH - norm * plotH;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
      }
      plotLine(computeVals, cMin, cMax, "#22d3ee");
      plotLine(cacheVals, kMin, kMax, "#f59e0b");

      // current-T marker
      const x = xForT(T);
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotH); ctx.stroke();
      ctx.setLineDash([]);

      // x-axis labels
      ctx.fillStyle = textColor; ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "center";
      [128, 1000, 10000, 100000, 1000000].forEach(t => {
        if (t < T_MIN || t > T_MAX) return;
        const xx = xForT(t);
        ctx.fillText(t >= 1e6 ? "1M" : t >= 1e3 ? (t / 1e3) + "K" : String(t), xx, h - 8);
      });
    }

    function updateReadouts() {
      const T = tFromSlider(state.slider);
      el("bills-t-val").textContent = fmtCompact(T) + " tokens";
      el("bills-u-val").textContent = state.users;
      el("bills-scores").textContent = fmtCompact(T * T);
      el("bills-cache-one").textContent = fmtBytes(kvCacheBytes(T));
      el("bills-cache-all").textContent = fmtBytes(kvCacheBytes(T) * state.users);
    }

    el("bills-t-slider").addEventListener("input", (e) => { state.slider = +e.target.value; draw(); updateReadouts(); });
    el("bills-u-slider").addEventListener("input", (e) => { state.users = +e.target.value; updateReadouts(); });
    window.addEventListener("resize", draw);

    draw(); updateReadouts();
  }

  /* ============================================================
     Embedded mini-widgets inside timeline cards
     ============================================================ */

  function initRopeWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const state = { distance: 6, ropeOn: true, basePos: 0 };
    const OMEGA = 0.28;

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">🌀</span>RoPE — relative angle vs. distance</div>
        <div style="display:grid;grid-template-columns:180px 1fr;gap:18px;align-items:center;">
          <canvas id="rope-canvas" width="180" height="180" style="width:100%;max-width:180px;"></canvas>
          <div>
            <div class="mw-controls">
              <button class="mw-btn" id="rope-toggle">RoPE: <b id="rope-toggle-label">ON</b></button>
              <button class="mw-btn" id="rope-shift">Shift both +5 (basePos)</button>
            </div>
            <div class="mw-slider-wrap">
              <span class="mw-slider-label">Distance between query &amp; key (i − j) <b id="rope-dist-val"></b></span>
              <input type="range" id="rope-dist-slider" min="0" max="20" value="${state.distance}" />
            </div>
            <div class="mw-readout" id="rope-readout"></div>
          </div>
        </div>
      </div>`;

    const canvas = el("rope-canvas");
    const ctx = canvas.getContext("2d");

    function draw() {
      const angleQ = state.ropeOn ? state.basePos * OMEGA : 0;
      const angleK = state.ropeOn ? (state.basePos + state.distance) * OMEGA : 0;
      const score = Math.cos(angleK - angleQ);

      const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 14;
      const isDark = document.documentElement.getAttribute("data-theme") !== "light";
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)";
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

      function arrow(angle, color, label) {
        const x = cx + r * Math.cos(angle), y = cy - r * Math.sin(angle);
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText(label, x + (x > cx ? 6 : -18), y + (y > cy ? 12 : -6));
      }
      arrow(angleQ, "#6366f1", "q");
      arrow(angleK, "#f43f5e", "k");
    }

    function readout() {
      const angleDiff = state.ropeOn ? state.distance * OMEGA : 0;
      const score = Math.cos(angleDiff);
      el("rope-readout").innerHTML = state.ropeOn
        ? `Relative angle = distance × ω = <b>${angleDiff.toFixed(2)} rad</b> → score = cos(Δθ) = <b class="good">${score.toFixed(3)}</b>. Shift both forward and this number won't move — only the <em>gap</em> matters.`
        : `RoPE off: both arrows stay fixed at 0° no matter what the distance slider says → score is always <b class="bad">1.000</b>. The model has no way to tell a near token from a far one from this score alone.`;
    }

    function render() { draw(); readout(); }

    el("rope-toggle").addEventListener("click", () => {
      state.ropeOn = !state.ropeOn;
      el("rope-toggle-label").textContent = state.ropeOn ? "ON" : "OFF";
      render();
    });
    el("rope-shift").addEventListener("click", () => { state.basePos = (state.basePos + 5) % 40; render(); });
    el("rope-dist-slider").addEventListener("input", (e) => {
      state.distance = +e.target.value;
      el("rope-dist-val").textContent = state.distance;
      render();
    });
    el("rope-dist-val").textContent = state.distance;
    render();
  }

  function initDeltaWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const state = { oldVal: 30, newVal: 70 };

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">✏️</span>Naive add vs. the delta rule</div>
        <div class="mw-controls">
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">State currently returns <b id="delta-old-val"></b> for key A</span>
            <input type="range" id="delta-old-slider" min="0" max="100" value="${state.oldVal}" />
          </div>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">It should now return <b id="delta-new-val"></b></span>
            <input type="range" id="delta-new-slider" min="0" max="100" value="${state.newVal}" />
          </div>
        </div>
        <div class="delta-demo">
          <div class="delta-box wrong">
            <h5>Naive additive write</h5>
            <div class="eq" id="delta-wrong-eq"></div>
          </div>
          <div class="delta-box right">
            <h5>Delta rule</h5>
            <div class="eq" id="delta-right-eq"></div>
          </div>
        </div>
      </div>`;

    function render() {
      const { oldVal, newVal } = state;
      const naive = oldVal + newVal;
      const delta = newVal - oldVal;
      const corrected = oldVal + delta;
      el("delta-old-val").textContent = oldVal;
      el("delta-new-val").textContent = newVal;
      el("delta-wrong-eq").innerHTML = `${oldVal} + ${newVal}<br/>= <span class="result">${naive}</span> ✗ not ${newVal}`;
      el("delta-right-eq").innerHTML = `delta = ${newVal} − ${oldVal} = ${delta}<br/>${oldVal} + ${delta}<br/>= <span class="result">${corrected}</span> ✓ exactly ${newVal}`;
    }

    el("delta-old-slider").addEventListener("input", (e) => { state.oldVal = +e.target.value; render(); });
    el("delta-new-slider").addEventListener("input", (e) => { state.newVal = +e.target.value; render(); });
    render();
  }

  function initSinksWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const N = 20;
    const state = { pos: 7, windowSize: 6, sinkCount: 0 };

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">🧷</span>Sliding-window eviction — with and without sinks</div>
        <div class="sw-strip" id="sw-strip"></div>
        <div class="mw-controls">
          <button class="mw-btn" id="sw-advance">Generate next token →</button>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">Window size <b id="sw-win-val"></b></span>
            <input type="range" id="sw-win-slider" min="3" max="12" value="${state.windowSize}" />
          </div>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">Sink tokens kept <b id="sw-sink-val"></b></span>
            <input type="range" id="sw-sink-slider" min="0" max="3" value="${state.sinkCount}" />
          </div>
        </div>
        <div class="mw-readout" id="sw-readout"></div>
      </div>`;

    function render() {
      const strip = el("sw-strip");
      let html = "";
      for (let i = 0; i < N; i++) {
        let cls = "";
        if (i > state.pos) cls = "";
        else {
          const inWindow = i > state.pos - state.windowSize && i <= state.pos;
          const isSink = state.sinkCount > 0 && i < state.sinkCount;
          if (isSink) cls = "is-sink";
          else if (inWindow) cls = "in-window";
          else cls = "evicted";
        }
        const style = i > state.pos ? "opacity:0.25;" : "";
        html += `<div class="sw-tok ${cls}" style="${style}">t${i}</div>`;
      }
      strip.innerHTML = html;

      el("sw-win-val").textContent = state.windowSize;
      el("sw-sink-val").textContent = state.sinkCount;

      const evictingSinceStart = state.pos - state.windowSize >= 0;
      const readout = el("sw-readout");
      if (!evictingSinceStart) {
        readout.innerHTML = `Kept so far: <b>${state.pos + 1}</b> tokens, none evicted yet — cache is still smaller than full history but nothing's been dropped.`;
      } else if (state.sinkCount === 0) {
        readout.innerHTML = `<span class="bad">⚠ Unstable.</span> Tokens t0–t${state.pos - state.windowSize} have been evicted, including the very first tokens. In real streaming models this is exactly where perplexity spikes.`;
      } else {
        readout.innerHTML = `<span class="good">✅ Stable.</span> The first <b>${state.sinkCount}</b> token(s) are pinned as attention sinks and never evicted, even though t${state.sinkCount}–t${Math.max(state.sinkCount, state.pos - state.windowSize)} were. Cache stays fixed at ${state.windowSize + state.sinkCount} tokens.`;
      }
    }

    el("sw-advance").addEventListener("click", () => { state.pos = Math.min(N - 1, state.pos + 1); render(); });
    el("sw-win-slider").addEventListener("input", (e) => { state.windowSize = +e.target.value; render(); });
    el("sw-sink-slider").addEventListener("input", (e) => { state.sinkCount = +e.target.value; render(); });
    render();
  }

  function initGqaWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const NUM_HEADS = 8;
    const state = { groups: 2 };
    const palette = ["#6366f1", "#22d3ee", "#f59e0b", "#f43f5e", "#10b981", "#a78bfa", "#06b6d4", "#94a3b8"];

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">🧩</span>Head-sharing: MHA → GQA → MQA</div>
        <div class="mw-controls">
          <button class="mw-btn" data-preset="8">MHA (8 groups)</button>
          <button class="mw-btn" data-preset="2">GQA (2 groups)</button>
          <button class="mw-btn" data-preset="1">MQA (1 group)</button>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">KV groups <b id="gqa-groups-val"></b></span>
            <input type="range" id="gqa-groups-slider" min="1" max="8" value="${state.groups}" />
          </div>
        </div>
        <div class="gqa-diagram" id="gqa-diagram"></div>
        <div class="mw-readout" id="gqa-readout"></div>
      </div>`;

    function render() {
      const g = state.groups;
      const headsPerGroup = NUM_HEADS / g;
      let qrow = "", krow = "";
      for (let i = 0; i < NUM_HEADS; i++) {
        const groupIdx = Math.floor(i / (NUM_HEADS / g));
        const color = palette[groupIdx % palette.length];
        qrow += `<div class="gqa-qhead" style="border-color:${color};color:${color};">Q${i}</div>`;
      }
      for (let gi = 0; gi < g; gi++) {
        const color = palette[gi % palette.length];
        krow += `<div class="gqa-khead" style="border-color:${color};color:${color};background:${color}22;">KV${gi}</div>`;
      }
      el("gqa-diagram").innerHTML = `<div class="gqa-qrow">${qrow}</div><div class="gqa-krow">${krow}</div>`;
      el("gqa-groups-val").textContent = g + (g === NUM_HEADS ? " (MHA)" : g === 1 ? " (MQA)" : " (GQA)");

      const reduction = NUM_HEADS / g;
      el("gqa-readout").innerHTML = g === NUM_HEADS
        ? `Every query head keeps its own K/V head — full quality, full cache (baseline).`
        : `Each K/V head is now shared by <b>${headsPerGroup}</b> query head(s) → KV cache is <b class="good">${reduction}× smaller</b> than MHA.`;
    }

    mount.querySelectorAll("[data-preset]").forEach(b => {
      b.addEventListener("click", () => {
        state.groups = +b.dataset.preset;
        el("gqa-groups-slider").value = state.groups;
        render();
      });
    });
    el("gqa-groups-slider").addEventListener("input", (e) => { state.groups = +e.target.value; render(); });
    render();
  }

  function initMlaWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const HEAD_DIM = 128;
    const MHA_WIDTH = 8 * HEAD_DIM;   // 1024
    const GQA_WIDTH = 2 * HEAD_DIM;   // 256
    const state = { latent: 96 };

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">🗜️</span>Per-token cache width — illustrative, not DeepSeek's exact reported numbers</div>
        <div class="mw-slider-wrap" style="margin-bottom:14px;">
          <span class="mw-slider-label">MLA latent width <b id="mla-latent-val"></b></span>
          <input type="range" id="mla-latent-slider" min="16" max="512" value="${state.latent}" />
        </div>
        <div id="mla-bars"></div>
        <div class="mw-readout" id="mla-readout"></div>
      </div>`;

    function bar(label, width, max, color) {
      const pct = Math.max(2, (width / max) * 100);
      return `<div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);margin-bottom:3px;">
          <span>${label}</span><span class="mono">${width} units/token</span>
        </div>
        <div class="cm-bar-track" style="height:14px;"><div class="cm-bar-fill" style="width:${pct}%;background:${color};"></div></div>
      </div>`;
    }

    function render() {
      const max = MHA_WIDTH;
      el("mla-bars").innerHTML =
        bar("MHA (8 heads × 128 dim)", MHA_WIDTH, max, "linear-gradient(90deg,#f43f5e,#fb7185)") +
        bar("GQA (2 groups × 128 dim)", GQA_WIDTH, max, "linear-gradient(90deg,#f59e0b,#fbbf24)") +
        bar("MLA (compressed latent)", state.latent, max, "linear-gradient(90deg,#10b981,#06b6d4)");
      el("mla-latent-val").textContent = state.latent;
      el("mla-readout").innerHTML = `At this latent width, MLA's per-token cache is <b class="good">${(MHA_WIDTH / state.latent).toFixed(1)}×</b> smaller than MHA and <b class="good">${(GQA_WIDTH / state.latent).toFixed(1)}×</b> smaller than this GQA config — DeepSeek-V2 reports both a smaller cache <em>and</em> better benchmark scores than MHA at their real configuration.`;
    }

    el("mla-latent-slider").addEventListener("input", (e) => { state.latent = +e.target.value; render(); });
    render();
  }

  function initNsaWidget(nodeId) {
    const mount = el("mini-widget-" + nodeId);
    if (!mount) return;
    const NUM_BLOCKS = 8, TOKENS_PER_BLOCK = 4;
    const state = { topk: 3 };

    mount.innerHTML = `
      <div class="mini-widget">
        <div class="mini-widget-title"><span class="mw-icon">🔎</span>Compress, then select top-k blocks</div>
        <div class="mw-slider-wrap" style="margin-bottom:12px;">
          <span class="mw-slider-label">Top-k blocks re-read at full resolution <b id="nsa-k-val"></b></span>
          <input type="range" id="nsa-k-slider" min="1" max="${NUM_BLOCKS}" value="${state.topk}" />
        </div>
        <div id="nsa-blocks" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
        <div class="mw-readout" id="nsa-readout"></div>
      </div>`;

    function render() {
      let html = "";
      for (let b = 0; b < NUM_BLOCKS; b++) {
        const selected = b < state.topk;
        html += `<div style="display:flex;flex-direction:column;gap:3px;padding:6px;border-radius:8px;border:1px solid ${selected ? "var(--indigo)" : "var(--border)"};background:${selected ? "rgba(99,102,241,0.12)" : "var(--bg-card-alt)"};">
          <div style="font-size:9px;color:${selected ? "var(--indigo)" : "var(--text-muted)"};text-align:center;font-weight:700;">block ${b}</div>
          <div style="display:flex;gap:2px;">${Array.from({ length: TOKENS_PER_BLOCK }).map(() => `<div style="width:8px;height:8px;border-radius:2px;background:${selected ? "var(--indigo)" : "var(--text-muted)"};opacity:${selected ? 1 : 0.35};"></div>`).join("")}</div>
        </div>`;
      }
      el("nsa-blocks").innerHTML = html;
      el("nsa-k-val").textContent = state.topk + " / " + NUM_BLOCKS;
      const totalTokens = NUM_BLOCKS * TOKENS_PER_BLOCK;
      const readTokens = state.topk * TOKENS_PER_BLOCK;
      el("nsa-readout").innerHTML = `${totalTokens} tokens compress into ${NUM_BLOCKS} block summaries (always cheap). Only the top-<b>${state.topk}</b> blocks — <b>${readTokens}</b> real tokens — get re-read at full resolution by the selected-attention branch; a small low-rank indexer picks them, so scoring the candidates stays cheap too.`;
    }

    el("nsa-k-slider").addEventListener("input", (e) => { state.topk = +e.target.value; render(); });
    render();
  }

  /* ── Reading progress bar ────────────────────────────────── */
  function initReadingProgress() {
    const bar = el("reading-progress");
    if (!bar) return;
    function update() {
      const doc = document.documentElement;
      const scrollable = doc.scrollHeight - doc.clientHeight;
      const pct = scrollable > 0 ? (doc.scrollTop / scrollable) * 100 : 0;
      bar.style.width = pct + "%";
    }
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  /* ── Scroll-reveal for narrative pacing ─────────────────── */
  function initScrollReveal() {
    const targets = document.querySelectorAll(".reveal");
    if (!targets.length) return;
    if (typeof IntersectionObserver === "undefined") {
      targets.forEach(t => t.classList.add("in-view"));
      return;
    }
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          obs.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    targets.forEach(t => obs.observe(t));
  }

  /* ── Hero canvas: a small live attention graph, purely decorative ── */
  function initHeroCanvas() {
    const canvas = el("hero-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const N = 14;
    let nodes = [];

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height;
    }
    function seed() {
      nodes = Array.from({ length: N }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
      }));
    }
    function step(t) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = nodes[i], b = nodes[j];
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          const maxD = Math.min(canvas.width, canvas.height) * 0.32;
          if (d < maxD) {
            const w = (1 - d / maxD) * (0.55 + 0.45 * Math.sin(t / 900 + i + j));
            if (w > 0.08) {
              ctx.strokeStyle = `rgba(167,139,250,${(w * 0.5).toFixed(2)})`;
              ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            }
          }
        }
      }
      nodes.forEach(n => {
        ctx.beginPath(); ctx.arc(n.x, n.y, 2.4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244,63,94,0.55)"; ctx.fill();
      });
      requestAnimationFrame(step);
    }
    resize(); seed();
    window.addEventListener("resize", () => { resize(); seed(); });
    requestAnimationFrame(step);
  }

  /* ── Boot ────────────────────────────────────────────────── */
  document.addEventListener("DOMContentLoaded", function () {
    renderHeroStats();
    renderStaticLegend("overview-thread-legend");
    renderInteractiveLegend("timeline-thread-legend", applyTimelineFilter);
    renderTimeline();
    renderCheatSheet();
    renderReferenceList();
    initCoreMechanismWidget();
    initTwoBillsWidget();
    initHeroCanvas();
    initReadingProgress();
    initScrollReveal();
  });
})();
