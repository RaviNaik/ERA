// js/corpus-widget.js — Corpus overview section: stat cards, chart, viewer
import { LANG_META, ASSET_BASE } from './data.js';

const LANGS = Object.keys(LANG_META);

// ── Hero score quick-cards ────────────────────────────────────
export function renderHeroScores(experiments) {
  const el = document.getElementById('hero-scores');
  if (!el) return;
  const colors = { step1: '#e07c8c', step2: '#5eba80', step3a: '#d4902a', step3b: '#5080d0' };
  const scoreColor = s => s >= 1200 ? '#5eba80' : s >= 800 ? '#d4902a' : '#e07c8c';
  el.innerHTML = experiments.map(exp => `
    <a href="#${exp.id}" class="hero-score-card" style="border-top:3px solid ${exp.accent}">
      <div class="hsc-step" style="color:${exp.accent}">${exp.step}</div>
      <div class="hsc-name">${exp.name}</div>
      <div class="hsc-score" style="color:${scoreColor(exp.score)}">${exp.score.toFixed(0)}</div>
    </a>`).join('');
}

// ── Language stat cards ───────────────────────────────────────
export function renderLangCards() {
  const el = document.getElementById('lang-cards');
  if (!el) return;
  el.innerHTML = LANGS.map(lang => {
    const m = LANG_META[lang];
    return `
    <div class="lang-card" style="background:${m.light};border-color:${m.soft}">
      <div class="lang-card-flag">${m.flag}</div>
      <div class="lang-card-name" style="color:${m.color}">${m.name}</div>
      <div class="lang-card-title">${m.title}</div>
      <div class="lang-card-stats">
        <div class="lcs-row"><span class="lcs-key">Characters</span><span class="lcs-val">${m.chars.toLocaleString()}</span></div>
        <div class="lcs-row"><span class="lcs-key">Words</span><span class="lcs-val">${m.words.toLocaleString()}</span></div>
        <div class="lcs-row"><span class="lcs-key">Avg word length</span><span class="lcs-val">${m.avgLen} ch</span></div>
      </div>
    </div>`;
  }).join('');
}

// ── Corpus bar chart ──────────────────────────────────────────
export function renderCorpusChart() {
  const ctx = document.getElementById('corpusChart');
  if (!ctx) return;
  const meta = Object.values(LANG_META);
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: meta.map(m => m.name),
      datasets: [
        {
          label: 'Words',
          data: meta.map(m => m.words),
          backgroundColor: meta.map(m => m.soft),
          borderColor:     meta.map(m => m.color),
          borderWidth: 2, borderRadius: 8,
        },
        {
          label: 'Chars',
          data: meta.map(m => m.chars),
          backgroundColor: meta.map(m => m.color + '44'),
          borderColor:     meta.map(m => m.color),
          borderWidth: 1.5, borderRadius: 8,
          yAxisID: 'y2',
        }
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { position: 'top', labels: { usePointStyle: true, font: { family: 'Inter', size: 11 } } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: 'Inter' } } },
        y:  { title: { display: true, text: 'Words', font: { family: 'Inter', size: 11 } }, grid: { color: 'rgba(0,0,0,.05)' } },
        y2: { position: 'right', title: { display: true, text: 'Characters', font: { family: 'Inter', size: 11 } }, grid: { display: false } },
      },
    },
  });
}

// ── Corpus viewer (tabs + preview + download) ─────────────────
export function initCorpusViewer() {
  const tabsEl    = document.getElementById('corpus-tabs');
  const previewEl = document.getElementById('corpus-preview');
  const actionsEl = document.getElementById('corpus-actions');
  if (!tabsEl) return;

  let activeLang = null;

  tabsEl.innerHTML = LANGS.map(lang => {
    const m = LANG_META[lang];
    return `<button class="corpus-tab" data-lang="${lang}"
      style="border-color:${m.color};color:${m.color}"
      onclick="window._selectCorpusLang('${lang}')">
      ${m.flag} ${m.name}
    </button>`;
  }).join('');

  window._selectCorpusLang = async (lang) => {
    if (activeLang === lang) return;
    activeLang = lang;
    const m = LANG_META[lang];

    // Update tab styles
    tabsEl.querySelectorAll('.corpus-tab').forEach(t => {
      const tl = t.dataset.lang;
      const tm = LANG_META[tl];
      if (tl === lang) {
        t.style.background = tm.color;
        t.style.color = 'white';
        t.classList.add('active');
      } else {
        t.style.background = 'rgba(255,255,255,0.5)';
        t.style.color = tm.color;
        t.classList.remove('active');
      }
    });

    previewEl.innerHTML = `<div class="corpus-loading">Loading ${m.name} corpus…</div>`;
    actionsEl.innerHTML = '';

    try {
      const url = `${ASSET_BASE}/data/${m.file}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(resp.statusText);
      const text = await resp.text();
      const preview = text.slice(0, 800).trim();
      const wordCount = text.trim().split(/\s+/).length;
      previewEl.textContent = preview + (text.length > 800 ? '\n\n[… truncated — showing first 800 chars]' : '');
      actionsEl.innerHTML = `
        <span class="text-muted" style="font-size:.8rem">${wordCount.toLocaleString()} words · ${text.length.toLocaleString()} chars</span>
        <a href="${url}" download="${m.file}" class="btn btn-outline btn-sm">⬇ Download ${m.name} Text</a>`;
    } catch (e) {
      previewEl.innerHTML = `<div class="corpus-loading" style="color:#e07c8c">Failed to load: ${e.message}</div>`;
    }
  };

  // Auto-select English
  window._selectCorpusLang('en');
}
