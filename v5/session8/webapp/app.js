/* ============================================================
   app.js — renders the hero stats, thread legend, chronological
   timeline, cheat sheet and reference list from data.js, and
   builds a dedicated visual explainer for every mechanism on the
   page, plus the narrative scaffolding: era interludes,
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
  function fmtCompact(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
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
  function isDark() { return document.documentElement.getAttribute("data-theme") !== "light"; }

  /* ── Hero stats ──────────────────────────────────────────── */
  function renderHeroStats() {
    const total = TIMELINE.length;
    const communitySourced = TIMELINE.filter(n => n.isCommunitySource).length;
    if (el("stat-count")) el("stat-count").textContent = total;
    if (el("stat-range")) el("stat-range").textContent = "2017 → 2025";
    if (el("stat-sourced")) el("stat-sourced").textContent = total + " / " + total;
    if (el("stat-flagged")) el("stat-flagged").textContent = communitySourced;
    if (el("stat-threads")) el("stat-threads").textContent = THREADS.length;
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

  /* "Pays down:" — the through-line the whole page hangs on. Every card
     shows which of the two bills (or the position problem) it moves. */
  function meterRowHTML(node) {
    const items = node.threads.map(id => {
      const t = threadById[id];
      if (!t) return "";
      return `<span class="tl-meter" style="--mc:${t.color}">
                <span class="tl-meter-dot"></span>${t.meter}
              </span>`;
    }).join("");
    return `<div class="tl-meters"><span class="tl-meters-label">Moves the needle on</span>${items}</div>`;
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
    if (node.isBonus) badges.push(`<span class="badge badge-purple">Not on the required list — included anyway</span>`);
    if (node.isCommunitySource) badges.push(`<span class="badge badge-yellow">Community post, not peer-reviewed</span>`);

    const nodeClasses = ["tl-node", "reveal"];
    if (node.isBaseline) nodeClasses.push("is-baseline");

    const dotColor = threadById[node.threads[0]] ? threadById[node.threads[0]].color : "var(--indigo)";
    const kicker = node.isBaseline ? "Where the story begins" : "The idea that answered it";

    return `${eraInterludeHTML(node)}
      <div class="${nodeClasses.join(" ")}" id="tl-${node.id}" data-threads="${node.threads.join(",")}">
        <div class="tl-dot" style="border-color:${dotColor}"></div>
        <div class="tl-card">
          <div class="tl-kicker-row">
            <span class="tl-date">${esc(node.dateDisplay)}</span>
            <div class="tl-threads">${threadTagsHTML(node.threads)}</div>
            <span class="tl-index">${index} / ${TIMELINE.length}</span>
          </div>
          <div class="tl-body">
            <div class="tl-block is-lede">
              <span class="tl-block-label">The situation it walked into</span>
              <span class="tl-block-text">${esc(node.problem)}</span>
            </div>

            <div class="tl-reveal">
              <div class="tl-reveal-kicker">${kicker} —</div>
              <h3 class="tl-reveal-title">${esc(node.title)}</h3>
              <div class="tl-tagline">${esc(node.tagline)}</div>
              ${badges.length ? `<div class="tl-badges">${badges.join(" ")}</div>` : ""}
            </div>

            ${node.intuition ? `<div class="tl-intuition">
              <span class="tl-intuition-label">In plain words</span>
              <span class="tl-intuition-text">${esc(node.intuition)}</span>
            </div>` : ""}

            ${node.diagram ? `<div class="tl-diagram" id="diagram-${node.id}"></div>` : ""}

            <div class="tl-block">
              <span class="tl-block-label">How it actually works</span>
              <span class="tl-block-text">${esc(node.mechanism)}</span>
            </div>

            <div class="tl-triad">
              <div class="tl-triad-item buy"><div class="tl-triad-label">✅ What it buys</div><div class="tl-triad-text">${esc(node.buys)}</div></div>
              <div class="tl-triad-item cost"><div class="tl-triad-label">⚠️ What it gives up</div><div class="tl-triad-text">${esc(node.costs)}</div></div>
              <div class="tl-triad-item choose"><div class="tl-triad-label">🎯 When you'd actually pick it</div><div class="tl-triad-text">${esc(node.chooseWhen)}</div></div>
            </div>
            ${meterRowHTML(node)}
            ${node.footnote ? `<div class="tl-footnote"><strong>Note —</strong> ${esc(node.footnote)}</div>` : ""}
            ${node.bridge ? `<div class="tl-bridge">${node.isEpilogue ? "" : "→ "}${esc(node.bridge)}</div>` : ""}
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
    container.innerHTML = TIMELINE.map(node => { i++; return cardHTML(node, i); }).join("");

    // Mount the per-mechanism visual explainer on each card.
    TIMELINE.forEach(node => {
      if (!node.diagram) return;
      const mount = el("diagram-" + node.id);
      const fn = DIAGRAMS[node.diagram];
      if (mount && fn) {
        try { fn(mount, node); }
        catch (e) { mount.innerHTML = `<div class="dg-fallback">(diagram unavailable)</div>`; }
      }
    });
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
     CHAPTER ONE — the mechanism itself, computed live on one
     sentence, one small idea at a time.
     Sentence deliberately different from the class notes.
     ============================================================ */
  function initCoreMechanismWidget() {
    const mounts = {
      qkv: el("concept-qkv"),
      scores: el("concept-scores"),
      scale: el("concept-scale"),
      mask: el("concept-mask"),
      softmax: el("concept-softmax"),
      output: el("concept-output"),
    };
    if (!mounts.qkv && !mounts.scores) return;

    const TOKENS = ["the", "bird", "fed", "its", "chicks"];
    // Hand-picked so "its" (pos 3) clearly resolves to "bird" (pos 1),
    // and "fed" (pos 2) also leans on "bird" — an intuitive result to read off.
    const Q = [[1,0,0,1],[0,1,1,0],[0,2,1,0],[0,2,1,0],[0,0,2,1]];
    const K = [[1,0,0,1],[0,2,1,0],[1,1,0,0],[0,1,1,1],[1,0,2,0]];
    const V = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]];
    const dk = 4, scaleDenom = Math.sqrt(dk);
    const N = TOKENS.length;

    const state = { maskOn: true };

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
      return `<div class="cm-qkv-legend">
          <span><span class="cm-chip q"></span>query — the question this word is asking</span>
          <span><span class="cm-chip k"></span>key — the name-tag it wears</span>
          <span><span class="cm-chip v"></span>value — the parcel it hands over if picked</span>
        </div>
        <div class="cm-tokens">${TOKENS.map((tok, i) => `
        <div class="cm-token-card">
          <div class="cm-token-name">${tok}</div>
          <div class="cm-token-row"><span class="cm-token-tag q">Q</span>${vecHTML(Q[i])}</div>
          <div class="cm-token-row"><span class="cm-token-tag k">K</span>${vecHTML(K[i])}</div>
          <div class="cm-token-row"><span class="cm-token-tag v">V</span>${vecHTML(V[i])}</div>
        </div>`).join("")}</div>`;
    }

    function matrixHTML(matrix, opts) {
      opts = opts || {};
      const rows = matrix.map((row, i) => {
        const cells = row.map((v, j) => {
          const isFuture = j > i;
          const masked = opts.showMask && state.maskOn && isFuture;
          const leak = opts.showMask && !state.maskOn && isFuture;
          const txt = masked ? "−∞" : (Number.isFinite(v) ? v.toFixed(opts.decimals ?? 2) : "−∞");
          let bg = "";
          if (!masked && Number.isFinite(v) && opts.heat) {
            const t = Math.max(0, Math.min(1, (v - opts.heat.min) / (opts.heat.max - opts.heat.min || 1)));
            bg = `background:rgba(99,102,241,${(0.06 + t * 0.4).toFixed(2)});`;
          }
          const isMax = opts.markRowMax && !masked && Number.isFinite(v) &&
                        v === Math.max(...row.filter(Number.isFinite));
          return `<div class="cm-cell ${masked ? "masked" : ""} ${leak ? "leak" : ""} ${isMax ? "rowmax" : ""}" style="${bg}">${txt}</div>`;
        }).join("");
        return `<div class="cm-matrix-line">
                  <span class="cm-rowlabel">${TOKENS[i]}</span>
                  <div class="cm-matrix-row" style="grid-template-columns:repeat(${row.length},1fr);">${cells}</div>
                </div>`;
      }).join("");
      return `<div class="cm-matrix">
                <div class="cm-axis"><span class="cm-rowlabel"></span>${TOKENS.map(t => `<span>${t}</span>`).join("")}</div>
                ${rows}
              </div>
              <div class="cm-axis-caption">rows = the word asking · columns = the word being looked at</div>`;
    }

    function renderQKV() { if (mounts.qkv) mounts.qkv.innerHTML = tokensPanelHTML(); }

    function renderScores(raw) {
      if (!mounts.scores) return;
      mounts.scores.innerHTML = matrixHTML(raw, {
        decimals: 0,
        heat: { min: Math.min(...raw.flat()), max: Math.max(...raw.flat()) },
      }) + `<p class="mw-readout">Read row <b>its</b>: its highest number by far is under <b>bird</b> — “its” is, in effect, asking “whose chicks?” and <b>bird</b>'s name-tag answers best.</p>`;
    }

    function renderScale(raw, scaled) {
      if (!mounts.scale) return;
      const rawMax = Math.max(...raw.flat());
      const scaledMax = Math.max(...scaled.flat());
      mounts.scale.innerHTML = matrixHTML(scaled, { heat: { min: Math.min(...scaled.flat()), max: scaledMax } }) +
        `<p class="mw-readout">Biggest raw score <b>${rawMax.toFixed(0)}</b> → biggest scaled score <b>${scaledMax.toFixed(2)}</b> (÷ √${dk} = ${scaleDenom.toFixed(0)}). The ranking of who-matches-whom is untouched — only the size of the numbers changed, so softmax next doesn't get overwhelmed.</p>`;
    }

    function renderMask(scaled) {
      if (!mounts.mask) return;
      mounts.mask.innerHTML = matrixHTML(scaled, { showMask: true }) +
        `<div class="cm-toggle-row">
          <button class="cm-switch ${state.maskOn ? "on" : ""}" id="cm-mask-switch" aria-pressed="${state.maskOn}"></button>
          <span class="cm-toggle-label">Causal mask ${state.maskOn ? "ON — the future is blocked" : "OFF — watch the rose-outlined cells"}</span>
        </div>
        <p class="mw-readout">${state.maskOn
          ? "Every hatched cell looks into the future (a later column than the row's own word). It's forced to −∞ before softmax ever runs, so “fed” can't cheat by peeking at “chicks”."
          : "Those cells aren't hatched anymore — they hold real numbers, about to receive real attention. During training that means the model gets to see the answer it's supposed to predict. That's the leak the mask exists to stop."}</p>`;
      const sw = el("cm-mask-switch");
      if (sw) sw.addEventListener("click", () => { state.maskOn = !state.maskOn; renderMaskDependent(); });
    }

    function renderSoftmax(weights) {
      if (!mounts.softmax) return;
      mounts.softmax.innerHTML = `<div class="cm-out">${weights.map((w, i) => `
          <div class="cm-softrow-block">
            <div class="cm-softrow-head">the word <strong>${TOKENS[i]}</strong> splits its attention like this:</div>
            ${w.map((wt, j) => `
              <div class="cm-out-row">
                <span class="cm-outlabel">${TOKENS[j]}</span>
                <div class="cm-bar-track"><div class="cm-bar-fill" style="width:${(wt * 100).toFixed(1)}%;"></div></div>
                <span class="cm-outpct">${(wt * 100).toFixed(0)}%</span>
              </div>`).join("")}
          </div>`).join("")}</div>
        <p class="mw-readout">${state.maskOn
          ? "Every row adds up to 100%. Blocked (future) words sit at exactly 0% — softmax literally cannot hand weight to −∞. Notice “its” pours most of its attention onto “bird”."
          : "Future words now hold real weight — the same leak from the previous step, now written as percentages."}</p>`;
    }

    function renderOutput(output) {
      if (!mounts.output) return;
      mounts.output.innerHTML = `<div class="cm-tokens cm-tokens-out">${output.map((o, i) => `
          <div class="cm-token-card">
            <div class="cm-token-name">${TOKENS[i]}</div>
            <div class="cm-token-row"><span class="cm-token-tag out">new</span>${vecHTML(o, 2)}</div>
          </div>`).join("")}</div>
        <p class="mw-readout">Each word's new vector is a blend of the <b>value</b> parcels, mixed by the percentages above. “its” now literally carries most of “bird” inside it — the pronoun and its referent have been physically linked.</p>`;
    }

    function renderMaskDependent() {
      const { scaled, weights, output } = computeAll();
      renderMask(scaled);
      renderSoftmax(weights);
      renderOutput(output);
    }

    const { raw, scaled } = computeAll();
    renderQKV();
    renderScores(raw);
    renderScale(raw, scaled);
    renderMaskDependent();
  }

  /* ============================================================
     CHAPTER TWO — the two bills, made concrete.
     Left: "compare every word to every word" grid, grows as T².
     Right: "remember every word" stack, grows as T.
     Framed explicitly as the scoreboard for the rest of the page.
     ============================================================ */
  function initTwoBillsWidget() {
    const root = el("two-bills-widget");
    if (!root) return;

    // small illustrative T for the concrete pictures
    const GRID_STEPS = [4, 8, 16, 32, 64];
    const state = { gi: 2, users: 8 };
    // real cache slider (log) for the "bill in GB" readout
    const T_MIN = 512, T_MAX = 1_000_000;

    function kvCacheBytes(T) {
      return 2 * REF_CONFIG.layers * REF_CONFIG.kvHeads * REF_CONFIG.headDim * T * REF_CONFIG.bytesPerNumber;
    }

    root.innerHTML = `
      <div class="bills-grid">
        <!-- BILL 1 : compute -->
        <div class="bill-col">
          <div class="bill-head"><span class="bill-tag cyan">Bill 1 · Compute</span> compare every word with every word</div>
          <div class="bill-visual" id="bill-compute-visual"></div>
          <div class="bill-number"><span id="bill-compute-count" class="cyan"></span> query·key scores</div>
          <p class="bill-note">Every new word adds a comparison against <em>all</em> the others. The count is T×T — double the text, quadruple the work.</p>
        </div>
        <!-- BILL 2 : memory -->
        <div class="bill-col">
          <div class="bill-head"><span class="bill-tag amber">Bill 2 · Memory</span> keep every earlier word's key + value</div>
          <div class="bill-visual" id="bill-memory-visual"></div>
          <div class="bill-number"><span id="bill-memory-count" class="amber"></span> saved key/value pairs</div>
          <p class="bill-note">During generation the model stores one K/V pair per earlier word so it never recomputes them — the <b>KV cache</b>. It grows in a straight line with T, and it's <em>private to each conversation</em>.</p>
        </div>
      </div>

      <div class="bills-slider-row">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">Sentence length T (for the pictures above) <b id="bill-t-val"></b></span>
          <input type="range" id="bill-t-slider" min="0" max="${GRID_STEPS.length - 1}" value="${state.gi}" step="1" />
        </div>
      </div>

      <div class="bills-real">
        <div class="bills-real-head">Now put real numbers on Bill 2, with a fairly ordinary model
          <span class="badge badge-slate">48 layers · 8 KV heads · head_dim 128 · bf16</span>
        </div>
        <div class="mw-controls">
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">Real context length <b id="bills-rt-val"></b></span>
            <input type="range" id="bills-rt-slider" min="0" max="100" value="58" />
          </div>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">Concurrent conversations on the server <b id="bills-u-val"></b></span>
            <input type="range" id="bills-u-slider" min="1" max="32" value="${state.users}" />
          </div>
        </div>
        <div class="bills-readouts">
          <div class="bills-readout"><div class="br-label">Q·K scores at this length</div><div class="br-val cyan" id="bills-scores"></div></div>
          <div class="bills-readout"><div class="br-label">KV cache · one conversation</div><div class="br-val amber" id="bills-cache-one"></div></div>
          <div class="bills-readout"><div class="br-label">KV cache · all conversations</div><div class="br-val amber" id="bills-cache-all"></div></div>
        </div>
      </div>

      <div class="bills-legend-callout">
        <span class="blc-icon">🧭</span>
        <div>
          <strong>This is the scoreboard for the rest of the page.</strong>
          Standard attention was not “bad.” It bought exact, all-to-all access and handed back exactly these two bills
          (plus: it can't tell word order — Chapter Three's third thread). Every single idea in the timeline below is
          someone pushing down <em>one</em> of these meters and paying for it somewhere else. The coloured tag on each
          card — <span class="blc-chip cyan">Compute</span> <span class="blc-chip amber">Memory</span>
          <span class="blc-chip purple">Position</span> — tells you which.
        </div>
      </div>
    `;

    function renderComputeVisual(T) {
      const wrap = el("bill-compute-visual");
      const showT = Math.min(T, 16); // cap the drawn grid for legibility
      let cells = "";
      for (let i = 0; i < showT; i++) {
        for (let j = 0; j < showT; j++) {
          const causal = j <= i;
          cells += `<div class="bcell ${causal ? "on" : "off"}"></div>`;
        }
      }
      const note = T > showT ? `<div class="bill-visual-cap">showing 16×16 of ${T}×${T}</div>` : "";
      wrap.innerHTML = `<div class="bgrid" style="grid-template-columns:repeat(${showT},1fr);">${cells}</div>${note}`;
    }

    function renderMemoryVisual(T) {
      const wrap = el("bill-memory-visual");
      const showT = Math.min(T, 32);
      let bars = "";
      for (let i = 0; i < showT; i++) bars += `<div class="bkv"></div>`;
      const note = T > showT ? `<div class="bill-visual-cap">showing 32 of ${T}</div>` : "";
      wrap.innerHTML = `<div class="bstack">${bars}</div>${note}`;
    }

    function renderPictures() {
      const T = GRID_STEPS[state.gi];
      el("bill-t-val").textContent = T + " words";
      renderComputeVisual(T);
      renderMemoryVisual(T);
      el("bill-compute-count").textContent = fmtCompact(T * T) + " ";
      el("bill-memory-count").textContent = T + " ";
    }

    function tFromSlider(s) {
      return Math.round(T_MIN * Math.pow(T_MAX / T_MIN, s / 100));
    }
    function renderReal() {
      const rt = tFromSlider(+el("bills-rt-slider").value);
      const users = +el("bills-u-slider").value;
      el("bills-rt-val").textContent = fmtCompact(rt) + " tokens";
      el("bills-u-val").textContent = users;
      el("bills-scores").textContent = fmtCompact(rt * rt);
      el("bills-cache-one").textContent = fmtBytes(kvCacheBytes(rt));
      el("bills-cache-all").textContent = fmtBytes(kvCacheBytes(rt) * users);
    }

    el("bill-t-slider").addEventListener("input", e => { state.gi = +e.target.value; renderPictures(); });
    el("bills-rt-slider").addEventListener("input", renderReal);
    el("bills-u-slider").addEventListener("input", renderReal);

    renderPictures();
    renderReal();
  }

  /* ============================================================
     DIAGRAM LIBRARY — one visual explainer per mechanism.
     Kept mostly static / lightly interactive; theme-safe via CSS.
     Signature: fn(mountEl, node)
     ============================================================ */

  function dgShell(title, icon, bodyHTML, footHTML) {
    return `<div class="dg">
      <div class="dg-title"><span class="dg-icon">${icon}</span>${title}</div>
      <div class="dg-body">${bodyHTML}</div>
      ${footHTML ? `<div class="dg-foot">${footHTML}</div>` : ""}
    </div>`;
  }

  /* — Baseline: the six-step pipeline as a strip — */
  function renderPipelineDiagram(mount) {
    const steps = [
      ["Q · Kᵀ", "every query meets every key"],
      ["scores", "one number per word-pair"],
      ["÷ √dₖ", "shrink to a sane range"],
      ["+ mask", "delete the future"],
      ["softmax", "scores → percentages"],
      ["Σ w·V", "blend the value parcels"],
    ];
    const body = `<div class="dg-pipe">${steps.map((s, i) => `
      <div class="dg-pipe-step">
        <div class="dg-pipe-box">${s[0]}</div>
        <div class="dg-pipe-cap">${s[1]}</div>
      </div>${i < steps.length - 1 ? `<div class="dg-pipe-arrow">→</div>` : ""}`).join("")}</div>`;
    mount.innerHTML = dgShell("The whole layer, six steps", "🎯", body,
      "Chapter One above runs all six of these live on a real sentence. Everything past this card changes one of these steps.");
  }

  /* — Sinusoidal: position fingerprints from stacked waves — */
  function renderSinusoidalDiagram(mount) {
    const W = 520, H = 150, freqs = [0.55, 0.28, 0.14, 0.06];
    const state = { pos: 12 };
    mount.innerHTML = dgShell("Each position → a fingerprint of wave heights", "🌊", `
      <svg class="dg-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" id="sin-svg"></svg>
      <div class="dg-controls">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">position <b id="sin-pos"></b></span>
          <input type="range" id="sin-slider" min="0" max="40" value="${state.pos}" />
        </div>
      </div>
      <div class="mw-readout" id="sin-readout"></div>`,
      "Fast waves pin down “exactly where”; slow waves say “roughly where”. Nearby positions share almost the same fingerprint; far ones don't.");

    const svg = mount.querySelector("#sin-svg");
    function draw() {
      const p = state.pos;
      const xFor = t => 20 + (t / 40) * (W - 90);
      let paths = "";
      freqs.forEach((f, k) => {
        const midY = 18 + k * 30;
        let d = "";
        for (let t = 0; t <= 40; t += 0.5) {
          const x = xFor(t), y = midY - Math.sin(t * f) * 11;
          d += (t === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1) + " ";
        }
        paths += `<path class="dg-wave" d="${d}"/>`;
        const cy = midY - Math.sin(p * f) * 11;
        paths += `<circle class="dg-wave-dot" cx="${xFor(p).toFixed(1)}" cy="${cy.toFixed(1)}" r="3.5"/>`;
        // fingerprint bar on the right — grows left or right from the centre
        const v = Math.sin(p * f);
        const bx = W - 58, bw = 44, ctr = bx + bw / 2, half = v * bw / 2;
        paths += `<rect class="dg-fp-bg" x="${bx}" y="${midY - 6}" width="${bw}" height="12" rx="2"/>`;
        paths += `<rect class="dg-fp" x="${Math.min(ctr, ctr + half).toFixed(1)}" y="${midY - 6}" width="${Math.abs(half).toFixed(1)}" height="12"/>`;
      });
      const lineX = xFor(p);
      svg.innerHTML = `<line class="dg-posln" x1="${lineX}" y1="4" x2="${lineX}" y2="${H - 22}"/>
        ${paths}
        <text class="dg-svg-lbl" x="${W - 58}" y="${H - 8}">fingerprint</text>
        <text class="dg-svg-lbl" x="20" y="${H - 8}">4 waves, slowest at the bottom</text>`;
      el("sin-pos").textContent = p;
      el("sin-readout").innerHTML = `Position <b>${p}</b> and position <b>${p + 1}</b> land almost on top of each other on every wave — that's how the model feels “adjacent”. Position <b>${p}</b> vs <b>${(p + 20) % 41}</b> look nothing alike.`;
    }
    mount.querySelector("#sin-slider").addEventListener("input", e => { state.pos = +e.target.value; draw(); });
    draw();
  }

  /* — Learned position table: rows, then a wall — */
  function renderLearnedTableDiagram(mount) {
    const ROWS = 7, COLS = 8;
    function cellColor(r, c) {
      const h = (Math.sin(r * 12.9898 + c * 78.233) * 43758.5453) % 1;
      const v = Math.abs(h);
      return `hsl(${Math.floor(200 + v * 120)}, 55%, ${isDark() ? 30 + v * 25 : 55 + v * 25}%)`;
    }
    let rows = "";
    for (let r = 0; r < ROWS; r++) {
      let cs = "";
      for (let c = 0; c < COLS; c++) cs += `<div class="ltab-cell" style="background:${cellColor(r, c)}"></div>`;
      rows += `<div class="ltab-row"><span class="ltab-idx">pos ${r}</span><div class="ltab-cells">${cs}</div></div>`;
    }
    const body = `<div class="ltab">${rows}
      <div class="ltab-row ltab-wall">
        <span class="ltab-idx">pos ${ROWS}</span>
        <div class="ltab-cells ltab-nowall">no row was ever allocated — hard wall</div>
      </div>
    </div>
    <button class="mw-btn" id="ltab-ask">Ask the model about position ${ROWS} →</button>
    <div class="mw-readout" id="ltab-out">Each row is one trainable vector, filled in by gradient descent. Fine — until the input is longer than the table.</div>`;
    mount.innerHTML = dgShell("A trainable vector per position — up to a fixed maximum", "🗄️", body,
      "Sinusoidal has a formula for any position. A table just stops.");
    mount.querySelector("#ltab-ask").addEventListener("click", () => {
      const wall = mount.querySelector(".ltab-wall");
      wall.classList.remove("shake"); void wall.offsetWidth; wall.classList.add("shake");
      el("ltab-out").innerHTML = `<b class="bad">Nothing to return.</b> The model has no vector for position ${ROWS}, so behaviour past the trained length isn't “degraded” — it's undefined.`;
    });
  }

  /* — Sparse / strided: a causal grid, full vs pattern — */
  function renderSparseGridDiagram(mount) {
    const T = 14, stride = 4, local = 1;
    const state = { mode: "sparse" };
    function count(mode) {
      let n = 0;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
        if (mode === "full") n++;
        else if ((i - j) <= local || j % stride === 0) n++;
      }
      return n;
    }
    function draw() {
      let cells = "";
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j > i) { cells += `<div class="spcell void"></div>`; continue; }
          const keep = state.mode === "full" || (i - j) <= local || j % stride === 0;
          cells += `<div class="spcell ${keep ? "on" : "dim"}"></div>`;
        }
      }
      mount.querySelector("#sp-grid").style.gridTemplateColumns = `repeat(${T},1fr)`;
      mount.querySelector("#sp-grid").innerHTML = cells;
      const full = count("full"), spar = count("sparse");
      mount.querySelector("#sp-readout").innerHTML = state.mode === "full"
        ? `Full causal attention: <b>${full}</b> query·key comparisons for just ${T} words.`
        : `Local band (±${local}) + every ${stride}ᵗʰ word: <b>${spar}</b> comparisons — <b class="good">${(full / spar).toFixed(1)}× fewer</b>. A message still crosses the whole sequence in ~2 hops through the pattern.`;
      mount.querySelectorAll("#sp-toggle .mw-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.mode === state.mode));
    }
    mount.innerHTML = dgShell("Who each word is allowed to look at", "🕸️", `
      <div class="dg-controls" id="sp-toggle">
        <button class="mw-btn" data-mode="full">Full attention</button>
        <button class="mw-btn" data-mode="sparse">Sparse pattern</button>
      </div>
      <div class="spgrid" id="sp-grid"></div>
      <div class="mw-readout" id="sp-readout"></div>`,
      "The pattern is fixed by the architecture, not chosen per input — cheap, but blind if the important word isn't on the grid.");
    mount.querySelectorAll("#sp-toggle .mw-btn").forEach(b =>
      b.addEventListener("click", () => { state.mode = b.dataset.mode; draw(); }));
    draw();
  }

  /* — Head sharing: MHA / GQA / MQA — */
  function renderHeadShareDiagram(mount, node) {
    const NUM_HEADS = 8;
    const cfg = node.diagramConfig || {};
    const state = { groups: cfg.preset || 2 };
    const palette = ["#6366f1", "#22d3ee", "#f59e0b", "#f43f5e", "#10b981", "#a78bfa", "#06b6d4", "#94a3b8"];

    mount.innerHTML = dgShell("Query heads (top) sharing stored key/value heads (bottom)", "🧩", `
      <div class="dg-controls">
        <button class="mw-btn" data-preset="8">MHA · 8</button>
        <button class="mw-btn" data-preset="2">GQA · 2</button>
        <button class="mw-btn" data-preset="1">MQA · 1</button>
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">KV groups stored <b id="hs-val"></b></span>
          <input type="range" id="hs-slider" min="1" max="8" value="${state.groups}" />
        </div>
      </div>
      <div class="hs-diagram" id="hs-diagram"></div>
      <div class="mw-readout" id="hs-readout"></div>`,
      "Fewer stored K/V heads = smaller cache. The queries stay separate, so they can still ask different questions.");

    function render() {
      const g = state.groups;
      const headsPerGroup = NUM_HEADS / g;
      let qrow = "", krow = "", links = "";
      for (let i = 0; i < NUM_HEADS; i++) {
        const gi = Math.floor(i / headsPerGroup);
        const color = palette[gi % palette.length];
        qrow += `<div class="hs-q" style="border-color:${color};color:${color};">Q${i}</div>`;
      }
      for (let gi = 0; gi < g; gi++) {
        const color = palette[gi % palette.length];
        krow += `<div class="hs-kv" style="border-color:${color};color:${color};background:${color}22;">KV${gi}</div>`;
      }
      el("hs-diagram").innerHTML = `<div class="hs-qrow">${qrow}</div>
        <div class="hs-link">${headsPerGroup > 1 ? `each KV shared by ${headsPerGroup} queries` : "one KV per query"}</div>
        <div class="hs-krow">${krow}</div>`;
      el("hs-val").textContent = g + (g === NUM_HEADS ? " · MHA" : g === 1 ? " · MQA" : " · GQA");
      const reduction = NUM_HEADS / g;
      el("hs-readout").innerHTML = g === NUM_HEADS
        ? `Every query head keeps its own K/V head — full quality, full cache. This is the baseline everything else shrinks.`
        : `Cache stores <b>${g}</b> K/V head(s) instead of ${NUM_HEADS} → <b class="good">${reduction}× smaller</b> KV cache. ${g === 1 ? "Maximum saving, biggest quality hit." : "Most of the saving, much less quality lost than MQA."}`;
      mount.querySelectorAll("[data-preset]").forEach(b =>
        b.classList.toggle("active", +b.dataset.preset === g));
    }
    mount.querySelectorAll("[data-preset]").forEach(b =>
      b.addEventListener("click", () => { state.groups = +b.dataset.preset; el("hs-slider").value = state.groups; render(); }));
    el("hs-slider").addEventListener("input", e => { state.groups = +e.target.value; render(); });
    render();
  }

  /* — Sliding window: band grows with depth — */
  function renderSlidingWindowDiagram(mount) {
    const T = 18, w = 2;
    const state = { layer: 1 };
    function draw() {
      const reach = w * state.layer;
      let cells = "";
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j > i) { cells += `<div class="swd-c void"></div>`; continue; }
          const inWin = (i - j) <= w;
          const inReach = (i - j) <= reach;
          const glob = j === 0;
          const cls = glob ? "glob" : inWin ? "win" : inReach ? "reach" : "far";
          cells += `<div class="swd-c ${cls}"></div>`;
        }
      }
      mount.querySelector("#swd-grid").style.gridTemplateColumns = `repeat(${T},1fr)`;
      mount.querySelector("#swd-grid").innerHTML = cells;
      el("swd-lay").textContent = state.layer;
      el("swd-readout").innerHTML = `In <b>one</b> layer each word sees only ±${w} neighbours (dark) plus the global token, column 0 (violet).
        Stack <b>${state.layer}</b> layers and information can travel ≈ <b>${reach}</b> positions (lighter band) — reach grows with depth even though no single layer is expensive.`;
    }
    mount.innerHTML = dgShell("A narrow window per layer, widening reach with depth", "🪟", `
      <div class="dg-controls">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">layers stacked <b id="swd-lay"></b></span>
          <input type="range" id="swd-slider" min="1" max="6" value="${state.layer}" />
        </div>
      </div>
      <div class="swd-grid" id="swd-grid"></div>
      <div class="swd-key">
        <span><i class="swd-c win"></i> direct window (±${w})</span>
        <span><i class="swd-c reach"></i> reachable through depth</span>
        <span><i class="swd-c glob"></i> global token</span>
      </div>
      <div class="mw-readout" id="swd-readout"></div>`,
      "Cost is T × window, not T². The price: within a layer, anything past the window is invisible.");
    mount.querySelector("#swd-slider").addEventListener("input", e => { state.layer = +e.target.value; draw(); });
    draw();
  }

  /* — Linear attention: growing list vs one fixed state — */
  function renderRunningStateDiagram(mount) {
    const state = { n: 3 };
    const MAXDRAW = 12;
    function draw() {
      const n = state.n;
      let pairs = "";
      for (let i = 0; i < Math.min(n, MAXDRAW); i++)
        pairs += `<div class="rs-pair">k<sub>${i + 1}</sub> v<sub>${i + 1}</sub></div>`;
      if (n > MAXDRAW) pairs += `<div class="rs-pair more">+${n - MAXDRAW} more…</div>`;
      mount.querySelector("#rs-list").innerHTML = pairs;
      mount.querySelector("#rs-list-count").textContent = n + (n === 1 ? " pair stored" : " pairs stored");
      mount.querySelector("#rs-state-count").textContent = n + " words folded in";
      el("rs-readout").innerHTML = `After <b>${n}</b> word${n === 1 ? "" : "s"}: exact softmax must keep <b>${n}</b> separate key/value pairs and revisit them for every new query.
        Linear attention keeps <b>one</b> d×d matrix — <b>S ← S + kₜ·vₜᵀ</b> — no matter how long the history gets.`;
    }
    mount.innerHTML = dgShell("Keep the whole transcript, or just a running summary?", "🧮", `
      <div class="rs-cols">
        <div class="rs-col exact">
          <div class="rs-col-head">Exact softmax — keep everything</div>
          <div class="rs-list" id="rs-list"></div>
          <div class="rs-col-foot" id="rs-list-count"></div>
        </div>
        <div class="rs-col linear">
          <div class="rs-col-head">Linear — one fixed state</div>
          <div class="rs-state">S<span>d × d</span></div>
          <div class="rs-col-foot" id="rs-state-count"></div>
        </div>
      </div>
      <div class="dg-controls">
        <button class="mw-btn" id="rs-add">+ 1 word</button>
        <button class="mw-btn" id="rs-add10">+ 10 words</button>
        <button class="mw-btn" id="rs-reset">reset</button>
      </div>
      <div class="mw-readout" id="rs-readout"></div>`,
      "Removing softmax is what lets the old key/value pairs be pre-combined before the query arrives. That's the whole trick — and the whole cost: the summary is lossy and can't be un-written.");
    mount.querySelector("#rs-add").addEventListener("click", () => { state.n++; draw(); });
    mount.querySelector("#rs-add10").addEventListener("click", () => { state.n += 10; draw(); });
    mount.querySelector("#rs-reset").addEventListener("click", () => { state.n = 1; draw(); });
    draw();
  }

  /* — RoPE: relative angle vs distance — */
  function renderRopeDiagram(mount) {
    const state = { distance: 6, ropeOn: true, basePos: 0 };
    const OMEGA = 0.28;
    mount.innerHTML = dgShell("Position as rotation — only the gap survives the dot product", "🌀", `
      <div class="rope-layout">
        <svg class="dg-svg rope-dial" viewBox="0 0 160 160" id="rope-svg"></svg>
        <div class="rope-side">
          <div class="dg-controls">
            <button class="mw-btn" id="rope-toggle">RoPE: <b id="rope-tl">ON</b></button>
            <button class="mw-btn" id="rope-shift">slide both +5 seats</button>
          </div>
          <div class="mw-slider-wrap">
            <span class="mw-slider-label">distance between the two words <b id="rope-dv"></b></span>
            <input type="range" id="rope-slider" min="0" max="20" value="${state.distance}" />
          </div>
          <div class="mw-readout" id="rope-readout"></div>
        </div>
      </div>`,
      "Slide both words forward together and the score doesn't move — only the distance between them does.");
    const svg = mount.querySelector("#rope-svg");
    function draw() {
      const aQ = state.ropeOn ? state.basePos * OMEGA : 0;
      const aK = state.ropeOn ? (state.basePos + state.distance) * OMEGA : 0;
      const cx = 80, cy = 80, r = 60;
      function arrow(a, cls, label) {
        const x = cx + r * Math.cos(a), y = cy - r * Math.sin(a);
        return `<line class="dg-arrow ${cls}" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>
                <circle class="dg-arrow-dot ${cls}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4"/>
                <text class="dg-svg-lbl" x="${(x + (x > cx ? 6 : -14)).toFixed(1)}" y="${(y + (y > cy ? 14 : -6)).toFixed(1)}">${label}</text>`;
      }
      svg.innerHTML = `<circle class="dg-dial" cx="${cx}" cy="${cy}" r="${r}"/>${arrow(aQ, "q", "q")}${arrow(aK, "k", "k")}`;
      const diff = state.ropeOn ? state.distance * OMEGA : 0;
      const score = Math.cos(diff);
      el("rope-dv").textContent = state.distance;
      el("rope-readout").innerHTML = state.ropeOn
        ? `Angle between arrows = distance × ω = <b>${diff.toFixed(2)} rad</b> → positional score = cos(Δθ) = <b class="good">${score.toFixed(3)}</b>. Slide both forward: this number won't budge.`
        : `RoPE off: both arrows stay at 0° whatever the slider says → score always <b class="bad">1.000</b>. Identical content at different distances is indistinguishable.`;
    }
    mount.querySelector("#rope-toggle").addEventListener("click", () => {
      state.ropeOn = !state.ropeOn; el("rope-tl").textContent = state.ropeOn ? "ON" : "OFF"; draw();
    });
    mount.querySelector("#rope-shift").addEventListener("click", () => { state.basePos = (state.basePos + 5) % 40; draw(); });
    mount.querySelector("#rope-slider").addEventListener("input", e => { state.distance = +e.target.value; draw(); });
    draw();
  }

  /* — ALiBi: raw scores minus a distance ramp — */
  function renderAlibiDiagram(mount) {
    const KEYS = 9;
    const raw = [0.9, 0.4, 1.1, 0.7, 1.3, 0.6, 1.0, 0.8, 1.2]; // arbitrary content match
    const state = { slope: 0.5 };
    function draw() {
      const m = state.slope;
      let rows = "";
      for (let d = 0; d < KEYS; d++) {
        const pen = m * (KEYS - 1 - d);        // key 0 is furthest from the query at the right
        const biased = raw[d] - pen;
        const w = Math.max(2, (biased + 4) / 8 * 100);
        rows += `<div class="alibi-row">
          <span class="alibi-lbl">${KEYS - 1 - d} back</span>
          <div class="alibi-track"><div class="alibi-fill" style="width:${w.toFixed(0)}%"></div></div>
          <span class="alibi-num">${raw[d].toFixed(1)} − ${pen.toFixed(1)} = <b>${biased.toFixed(1)}</b></span>
        </div>`;
      }
      mount.querySelector("#alibi-rows").innerHTML = rows;
      el("alibi-sv").textContent = m.toFixed(2);
      el("alibi-readout").innerHTML = `Head slope <b>${m.toFixed(2)}</b>: each step of distance costs a flat <b>${m.toFixed(2)}</b> off the score, <em>before</em> softmax. No position vectors exist anywhere — different heads just use different slopes.`;
    }
    mount.innerHTML = dgShell("Subtract a flat penalty for distance — that's the entire position scheme", "📉", `
      <div class="alibi-rows" id="alibi-rows"></div>
      <div class="dg-controls">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">this head's slope m <b id="alibi-sv"></b></span>
          <input type="range" id="alibi-slider" min="0" max="1" step="0.05" value="${state.slope}" />
        </div>
      </div>
      <div class="mw-readout" id="alibi-readout"></div>`,
      "Trained short, runs long with no fine-tuning. The catch: the penalty is a fixed opinion the model can't argue with.");
    mount.querySelector("#alibi-slider").addEventListener("input", e => { state.slope = +e.target.value; draw(); });
    draw();
  }

  /* — FlashAttention: memory hierarchy, no full matrix — */
  function renderFlashDiagram(mount) {
    const body = `<div class="flash-wrap">
      <div class="flash-hbm">
        <div class="flash-lbl">HBM · large, slow</div>
        <div class="flash-tiles">
          <span class="flash-tile q">Q</span><span class="flash-tile k">K</span><span class="flash-tile v">V</span>
        </div>
        <div class="flash-nomatrix">full T×T score matrix — <b>never written here</b> ✗</div>
      </div>
      <div class="flash-arrow">stream tiles ⇅</div>
      <div class="flash-sram">
        <div class="flash-lbl">SRAM · tiny, fast</div>
        <div class="flash-sram-inner">
          <span class="flash-block">Qᵢ</span><span class="flash-op">×</span><span class="flash-block">Kⱼᵀ</span>
          <span class="flash-op">→</span><span class="flash-run">running softmax<br/>(m, ℓ, O)</span>
        </div>
      </div>
    </div>`;
    mount.innerHTML = dgShell("Same math, kept out of slow memory", "⚡", body,
      "Bit-for-bit identical output to standard attention. Roughly the same number of multiplies — it just stops shuttling the giant matrix back and forth.");
  }

  /* — NTK-aware: uneven frequency stretch — */
  function renderFreqStretchDiagram(mount) {
    const N = 10;
    const state = { factor: 4 };
    function bars(kind) {
      let out = "";
      for (let i = 0; i < N; i++) {
        const base = 1 - i / N * 0.85;                 // fast (left) → slow (right)
        let scaled;
        if (kind === "uniform") scaled = base / state.factor;
        else {                                          // ntk: stretch slow ones hard, fast ones barely
          const localness = 1 - i / (N - 1);            // 1 at fast end
          const f = 1 + (state.factor - 1) * (1 - localness) ** 1.5;
          scaled = base / f;
        }
        out += `<div class="fs-bar" style="height:${(scaled * 46 + 4).toFixed(0)}px"></div>`;
      }
      return out;
    }
    function draw() {
      mount.querySelector("#fs-uniform").innerHTML = bars("uniform");
      mount.querySelector("#fs-ntk").innerHTML = bars("ntk");
      el("fs-fv").textContent = state.factor + "×";
      el("fs-readout").innerHTML = `Extending context by <b>${state.factor}×</b>. Uniform squeeze shrinks the fast, local frequencies just as hard as the slow ones — that's the precision the model uses to tell neighbours apart. NTK-aware leaves the left bars nearly full height.`;
    }
    mount.innerHTML = dgShell("Rotation speed across frequencies — fast/local on the left", "🎚️", `
      <div class="fs-block"><div class="fs-title">Uniform squeeze — everything ÷ factor</div><div class="fs-bars" id="fs-uniform"></div></div>
      <div class="fs-block"><div class="fs-title">NTK-aware — stretch slow bands, spare the fast ones</div><div class="fs-bars ntk" id="fs-ntk"></div></div>
      <div class="dg-controls">
        <div class="mw-slider-wrap">
          <span class="mw-slider-label">context extension factor <b id="fs-fv"></b></span>
          <input type="range" id="fs-slider" min="2" max="16" step="1" value="${state.factor}" />
        </div>
      </div>
      <div class="mw-readout" id="fs-readout"></div>`,
      "One global formula, no retraining. Still coarse: every frequency gets the same rule.");
    mount.querySelector("#fs-slider").addEventListener("input", e => { state.factor = +e.target.value; draw(); });
    draw();
  }

  /* — YaRN: three bands — */
  function renderYarnBandsDiagram(mount) {
    const body = `<div class="yarn-spectrum">
        <div class="yarn-band keep"><div class="yb-name">KEEP</div><div class="yb-desc">fastest / most local frequencies — left exactly as trained</div></div>
        <div class="yarn-band blend"><div class="yb-name">BLEND</div><div class="yb-desc">middle band — feathered smoothly between the two</div></div>
        <div class="yarn-band stretch"><div class="yb-name">STRETCH</div><div class="yb-desc">slowest / most global frequencies — scaled for the new length</div></div>
      </div>
      <div class="yarn-axis"><span>high frequency · local detail</span><span>low frequency · long range</span></div>
      <div class="yarn-temp">+ a small <b>softmax temperature</b> correction — stretching positions quietly changes how sharp attention is, so YaRN nudges it back</div>`;
    mount.innerHTML = dgShell("The same stretch, but split into three zones", "🧶", body,
      "The current default for extending a rotary model after training. More knobs than NTK, and still an extrapolation — the model never actually trained at the long length.");
  }

  /* — Attention sinks: sliding-window eviction sim — */
  function renderSinksDiagram(mount) {
    const NT = 22;
    const state = { pos: 8, windowSize: 6, sinkCount: 0 };
    function render() {
      let html = "";
      for (let i = 0; i < NT; i++) {
        let cls = "future", style = "opacity:.25;";
        if (i <= state.pos) {
          style = "";
          const inWindow = i > state.pos - state.windowSize && i <= state.pos;
          const isSink = state.sinkCount > 0 && i < state.sinkCount;
          cls = isSink ? "is-sink" : inWindow ? "in-window" : "evicted";
        }
        html += `<div class="sw-tok ${cls}" style="${style}">t${i}</div>`;
      }
      mount.querySelector("#sk-strip").innerHTML = html;
      el("sk-win").textContent = state.windowSize;
      el("sk-sink").textContent = state.sinkCount;
      const evicting = state.pos - state.windowSize >= 0;
      const out = el("sk-readout");
      if (!evicting) out.innerHTML = `Kept: <b>${state.pos + 1}</b> words, nothing evicted yet.`;
      else if (state.sinkCount === 0) out.innerHTML = `<b class="bad">⚠ Unstable.</b> t0–t${state.pos - state.windowSize} were evicted, including the very first words. This is exactly where a real streaming model's output degrades — softmax has nowhere to dump its leftover attention.`;
      else out.innerHTML = `<b class="good">✅ Stable.</b> The first <b>${state.sinkCount}</b> word(s) are pinned as sinks and never evicted. Cache stays fixed at ${state.windowSize + state.sinkCount} words, and attention has a home for its spare mass.`;
    }
    mount.innerHTML = dgShell("A fixed window, running forever — with and without pinned sinks", "🧷", `
      <div class="sw-strip" id="sk-strip"></div>
      <div class="dg-controls">
        <button class="mw-btn" id="sk-next">generate next word →</button>
        <div class="mw-slider-wrap"><span class="mw-slider-label">window <b id="sk-win"></b></span>
          <input type="range" id="sk-winsl" min="3" max="12" value="${state.windowSize}" /></div>
        <div class="mw-slider-wrap"><span class="mw-slider-label">pinned sinks <b id="sk-sink"></b></span>
          <input type="range" id="sk-sinksl" min="0" max="4" value="${state.sinkCount}" /></div>
      </div>
      <div class="mw-readout" id="sk-readout"></div>`,
      "The sinks are a release valve, not a memory — the evicted middle is genuinely gone.");
    mount.querySelector("#sk-next").addEventListener("click", () => { state.pos = Math.min(NT - 1, state.pos + 1); render(); });
    mount.querySelector("#sk-winsl").addEventListener("input", e => { state.windowSize = +e.target.value; render(); });
    mount.querySelector("#sk-sinksl").addEventListener("input", e => { state.sinkCount = +e.target.value; render(); });
    render();
  }

  /* — MLA: per-token cache width — */
  function renderMlaDiagram(mount) {
    const HEAD_DIM = 128, MHA = 8 * HEAD_DIM, GQA = 2 * HEAD_DIM;
    const state = { latent: 96 };
    function bar(label, width, color) {
      const pct = Math.max(2, (width / MHA) * 100);
      return `<div class="mla-barrow">
        <div class="mla-barlbl"><span>${label}</span><span class="mono">${width}/token</span></div>
        <div class="mla-track"><div class="mla-fill" style="width:${pct}%;background:${color};"></div></div>
      </div>`;
    }
    function render() {
      mount.querySelector("#mla-bars").innerHTML =
        bar("MHA — 8 heads × 128", MHA, "#f43f5e") +
        bar("GQA — 2 groups × 128", GQA, "#f59e0b") +
        bar("MLA — one compressed latent", state.latent, "#10b981");
      el("mla-lv").textContent = state.latent;
      el("mla-readout").innerHTML = `At this latent width MLA's per-token cache is <b class="good">${(MHA / state.latent).toFixed(1)}×</b> smaller than MHA and <b class="good">${(GQA / state.latent).toFixed(1)}×</b> smaller than this GQA. The full keys/values are rebuilt from the latent on the fly — DeepSeek-V2 reported a smaller cache <em>and</em> better scores than MHA at their real config.`;
    }
    mount.innerHTML = dgShell("What each word leaves behind in the cache", "🗜️", `
      <div class="mw-slider-wrap" style="margin-bottom:12px;">
        <span class="mw-slider-label">MLA latent width <b id="mla-lv"></b></span>
        <input type="range" id="mla-slider" min="16" max="512" value="${state.latent}" />
      </div>
      <div id="mla-bars"></div>
      <div class="mw-readout" id="mla-readout"></div>`,
      "Illustrative widths, not DeepSeek's exact reported numbers. Still linear in T — just a much smaller constant.");
    mount.querySelector("#mla-slider").addEventListener("input", e => { state.latent = +e.target.value; render(); });
    render();
  }

  /* — Delta rule: read, diff, write only the gap — */
  function renderDeltaDiagram(mount) {
    const state = { oldVal: 30, newVal: 70 };
    function render() {
      const { oldVal, newVal } = state;
      const naive = oldVal + newVal, delta = newVal - oldVal, corrected = oldVal + delta;
      el("dl-ov").textContent = oldVal;
      el("dl-nv").textContent = newVal;
      mount.querySelector("#dl-wrong").innerHTML = `${oldVal} + ${newVal}<br/>= <span class="result">${naive}</span> ✗ wanted ${newVal}`;
      mount.querySelector("#dl-right").innerHTML = `gap = ${newVal} − ${oldVal} = ${delta}<br/>${oldVal} + ${delta}<br/>= <span class="result">${corrected}</span> ✓ exactly ${newVal}`;
    }
    mount.innerHTML = dgShell("Updating one fact in a fixed-size memory", "✏️", `
      <div class="dg-controls">
        <div class="mw-slider-wrap"><span class="mw-slider-label">memory currently returns <b id="dl-ov"></b> for key A</span>
          <input type="range" id="dl-osl" min="0" max="100" value="${state.oldVal}" /></div>
        <div class="mw-slider-wrap"><span class="mw-slider-label">it should now return <b id="dl-nv"></b></span>
          <input type="range" id="dl-nsl" min="0" max="100" value="${state.newVal}" /></div>
      </div>
      <div class="delta-demo">
        <div class="delta-box wrong"><h5>Just add the new answer</h5><div class="eq" id="dl-wrong"></div></div>
        <div class="delta-box right"><h5>Delta rule — write only the gap</h5><div class="eq" id="dl-right"></div></div>
      </div>`,
      "A plain running total can only ever pile information up. This is how a fixed-size state learns to overwrite.");
    mount.querySelector("#dl-osl").addEventListener("input", e => { state.oldVal = +e.target.value; render(); });
    mount.querySelector("#dl-nsl").addEventListener("input", e => { state.newVal = +e.target.value; render(); });
    render();
  }

  /* — Gated DeltaNet: correction + a forget gate — */
  function renderGatedDeltaDiagram(mount) {
    const state = { gate: 0.7 };
    const STEPS = 8, START = 100;
    function render() {
      const g = state.gate;
      let bars = "";
      for (let s = 0; s <= STEPS; s++) {
        const v = START * Math.pow(g, s);
        bars += `<div class="gd-barwrap"><div class="gd-bar" style="height:${(v / START * 60 + 3).toFixed(0)}px"></div><span>${s}</span></div>`;
      }
      mount.querySelector("#gd-bars").innerHTML = bars;
      el("gd-gv").textContent = g.toFixed(2);
      el("gd-readout").innerHTML = `A fact stored long ago, no longer relevant. Each step the gate keeps only <b>${g.toFixed(2)}</b> of it, so after ${STEPS} steps it's down to <b>${(Math.pow(g, STEPS) * 100).toFixed(0)}%</b> of its original strength — <b>${g >= 0.99 ? "with the gate open it never fades (that's the delta rule alone)" : "it decays away on its own"}</b>. The delta-rule correction from the previous card still runs on top of this.`;
    }
    mount.innerHTML = dgShell("Add a per-step forget gate on top of the delta rule", "🎛️", `
      <div class="gd-bars" id="gd-bars"></div>
      <div class="gd-axis">steps since the fact was written →</div>
      <div class="dg-controls">
        <div class="mw-slider-wrap"><span class="mw-slider-label">retention gate g (per step) <b id="gd-gv"></b></span>
          <input type="range" id="gd-slider" min="0.4" max="1" step="0.02" value="${state.gate}" /></div>
      </div>
      <div class="mw-readout" id="gd-readout"></div>`,
      "Correcting a wrong fact and letting an irrelevant one fade are different skills. Gated DeltaNet does both in one update — but discarded information is still gone for good.");
    mount.querySelector("#gd-slider").addEventListener("input", e => { state.gate = +e.target.value; render(); });
    render();
  }

  /* — NSA: compress everything, re-read the top-k blocks — */
  function renderNsaDiagram(mount) {
    const NB = 8, TPB = 4;
    const state = { topk: 3 };
    function render() {
      let html = "";
      for (let b = 0; b < NB; b++) {
        const sel = b < state.topk;
        html += `<div class="nsa-block ${sel ? "sel" : ""}">
          <div class="nsa-block-lbl">block ${b}</div>
          <div class="nsa-toks">${Array.from({ length: TPB }).map(() => `<span class="${sel ? "on" : ""}"></span>`).join("")}</div>
          <div class="nsa-block-tag">${sel ? "re-read in full" : "summary only"}</div>
        </div>`;
      }
      mount.querySelector("#nsa-blocks").innerHTML = html;
      el("nsa-kv").textContent = state.topk + " / " + NB;
      el("nsa-readout").innerHTML = `All <b>${NB * TPB}</b> words compress into <b>${NB}</b> cheap block summaries (always scanned). A small low-rank scorer picks the top <b>${state.topk}</b> blocks — <b>${state.topk * TPB}</b> real words — to re-read at full detail. Plus a local window, always. All three paths trained together from step one.`;
    }
    mount.innerHTML = dgShell("Skim every block cheaply, re-read only the promising few", "🔎", `
      <div class="mw-slider-wrap" style="margin-bottom:12px;">
        <span class="mw-slider-label">top-k blocks re-read at full resolution <b id="nsa-kv"></b></span>
        <input type="range" id="nsa-slider" min="1" max="${NB}" value="${state.topk}" />
      </div>
      <div class="nsa-blocks" id="nsa-blocks"></div>
      <div class="mw-readout" id="nsa-readout"></div>`,
      "The old 2019 problem — “finding the good keys costs as much as reading them all” — is what the cheap scorer finally fixes. Cost: a useful word in an unpicked block is simply missed.");
    mount.querySelector("#nsa-slider").addEventListener("input", e => { state.topk = +e.target.value; render(); });
    render();
  }

  /* — DroPE: the training schedule — */
  function renderDropeScheduleDiagram(mount) {
    const state = { showWhy: false };
    function render() {
      mount.querySelector("#drope-why").innerHTML = state.showWhy
        ? `<div class="drope-whybox">Skipping rotation from step one makes early training measurably slower and worse — attention heads have no shortcut for developing directional bias. So rotation isn't optional early; it's a scaffold you remove once it's done its job.</div>`
        : "";
      mount.querySelector("#drope-whybtn").textContent = state.showWhy ? "hide" : "why not skip rotation from the start?";
    }
    const body = `<div class="drope-bar">
        <div class="drope-phase train" style="flex: 96">
          <div class="dp-name">train WITH rotary position</div>
          <div class="dp-sub">~96–99% of pretraining · rotation helps order-structure form fast</div>
        </div>
        <div class="drope-phase recal" style="flex: 6">
          <div class="dp-name">remove rotation<br/>+ short recalibration</div>
          <div class="dp-sub">≈0.5–3% of training cost · at the original short length · no long-sequence data</div>
        </div>
      </div>
      <div class="drope-out">→ serve far beyond the training length, with <b>no position mechanism at all</b> — nothing left to stretch, so nothing to warp</div>
      <div class="dg-controls"><button class="mw-btn" id="drope-whybtn"></button></div>
      <div id="drope-why"></div>`;
    mount.innerHTML = dgShell("Rotation as a scaffold: keep it to build, then take it away", "🪜", body,
      "Every other extension method on this page stretches the rotation and pays for it in the long-range frequencies. This one deletes the rotation so there's nothing to stretch.");
    mount.querySelector("#drope-whybtn").addEventListener("click", () => { state.showWhy = !state.showWhy; render(); });
    render();
  }

  const DIAGRAMS = {
    pipeline: renderPipelineDiagram,
    sinusoidal: renderSinusoidalDiagram,
    "learned-table": renderLearnedTableDiagram,
    "sparse-grid": renderSparseGridDiagram,
    "head-share": renderHeadShareDiagram,
    "sliding-window": renderSlidingWindowDiagram,
    "running-state": renderRunningStateDiagram,
    rope: renderRopeDiagram,
    alibi: renderAlibiDiagram,
    flash: renderFlashDiagram,
    "freq-stretch": renderFreqStretchDiagram,
    "yarn-bands": renderYarnBandsDiagram,
    sinks: renderSinksDiagram,
    mla: renderMlaDiagram,
    delta: renderDeltaDiagram,
    "gated-delta": renderGatedDeltaDiagram,
    nsa: renderNsaDiagram,
    "drope-schedule": renderDropeScheduleDiagram,
  };

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
    // threshold:0 — fire as soon as any pixel crosses the viewport. A
    // percentage threshold silently fails for elements taller than the
    // viewport (e.g. the timeline section), leaving them stuck hidden.
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) { entry.target.classList.add("in-view"); obs.unobserve(entry.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0 });
    targets.forEach(t => obs.observe(t));

    // Failsafe mirroring the observer with a plain scroll check, so a
    // hidden element can never get trapped at opacity:0 if the observer
    // misbehaves for any reason.
    function sweep() {
      const hidden = document.querySelectorAll(".reveal:not(.in-view)");
      if (!hidden.length) { window.removeEventListener("scroll", sweep); return; }
      hidden.forEach(t => {
        if (t.getBoundingClientRect().top < window.innerHeight * 0.92) t.classList.add("in-view");
      });
    }
    sweep();
    window.addEventListener("scroll", sweep, { passive: true });
  }

  /* ── Hero canvas: a small live attention graph, decorative ── */
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
        x: Math.random() * canvas.width, y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.15, vy: (Math.random() - 0.5) * 0.15,
      }));
    }
    function step(t) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
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
