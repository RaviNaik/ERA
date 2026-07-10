// js/experiment-widget.js — Renders one card per experiment
import { LANG_META, ASSET_BASE } from './data.js';

const LANGS = ['en', 'hi', 'te', 'kn'];
const MAX_RATIO = 8; // for ratio bar scaling

function scoreColor(score) {
  if (score >= 1200) return '#5eba80';
  if (score >= 900)  return '#5080d0';
  if (score >= 600)  return '#d4902a';
  return '#e07c8c';
}

function rankEmoji(rank) {
  return ['🥇','🥈','🥉','4️⃣'][rank - 1] ?? rank;
}

function ratioRow(lang, data) {
  const m = LANG_META[lang];
  const pct = Math.min((data.ratio / MAX_RATIO) * 100, 100).toFixed(1);
  return `
  <tr>
    <td>
      <span class="lang-pill" style="background:${m.light};color:${m.color}">
        ${m.name}
      </span>
    </td>
    <td class="num">${data.words.toLocaleString()}</td>
    <td class="num">${data.tokens.toLocaleString()}</td>
    <td>
      <div class="ratio-val">
        <span style="color:${m.color};font-weight:700">${data.ratio.toFixed(4)}</span>
        <div class="ratio-bar-wrap">
          <div class="ratio-bar" style="width:${pct}%;background:${m.color}"></div>
        </div>
      </div>
    </td>
  </tr>`;
}

function configGrid(config) {
  return `<div class="metric-grid">
    ${Object.entries(config).map(([k, v]) => `
    <div class="metric-box">
      <label>${k}</label>
      <input type="text" readonly value="${v}">
    </div>`).join('')}
  </div>`;
}

function perLangMerge(perLangTrained) {
  if (!perLangTrained) return '';
  return `
  <div style="margin-top:12px">
    <div class="card-label" style="margin-bottom:8px">Per-Language Tokens Trained</div>
    <div class="metric-grid" style="grid-template-columns:repeat(4,1fr)">
      ${Object.entries(perLangTrained).map(([lang, trained]) => {
        const m = LANG_META[lang];
        const pct = ((trained / 2500) * 100).toFixed(0);
        return `
        <div class="metric-box">
          <label style="color:${m.color}">${m.name}</label>
          <input type="text" readonly value="${trained.toLocaleString()} / 2,500">
          <div style="height:4px;background:${m.soft};border-radius:2px;margin-top:4px">
            <div style="height:4px;width:${pct}%;background:${m.color};border-radius:2px"></div>
          </div>
        </div>`;
      }).join('')}
    </div>
  </div>`;
}

export function renderExperiments(experiments) {
  const container = document.getElementById('exp-cards');
  if (!container) return;

  container.innerHTML = experiments.map(exp => `
  <article class="exp-card" id="${exp.id}">

    <div class="exp-header">
      <div class="exp-header-left">
        <div class="exp-step-badge" style="background:${exp.accent}">${exp.step}</div>
        <h2 class="exp-title">${exp.name}</h2>
        <p class="exp-desc">${exp.desc}</p>
      </div>
      <div class="score-badge">
        <div class="score-badge-label">Score</div>
        <div class="score-badge-val" style="color:${scoreColor(exp.score)}">${exp.score.toFixed(0)}</div>
        <div class="score-badge-rank">${rankEmoji(exp.rank)} Rank ${exp.rank}</div>
      </div>
    </div>

    <div class="exp-body">
      <!-- Config -->
      <div class="exp-config">
        <div class="exp-config-title">Configuration</div>
        ${configGrid(exp.config)}
        ${perLangMerge(exp.perLangTrained)}
      </div>

      <!-- Results -->
      <div class="exp-results">
        <div class="exp-results-title">Fertility Ratios (tokens ÷ words)</div>
        <table class="ratio-table">
          <thead>
            <tr>
              <th>Language</th>
              <th>Words</th>
              <th>Tokens</th>
              <th>Ratio (X)</th>
            </tr>
          </thead>
          <tbody>
            ${LANGS.map(lang => ratioRow(lang, exp.results[lang])).join('')}
          </tbody>
        </table>
        <div class="exp-chart-wrap">
          <canvas id="chart-${exp.id}" height="120"></canvas>
        </div>
      </div>
    </div>

    <!-- Score calculation -->
    <div class="score-calc">
      <div class="score-calc-title">Score Calculation</div>
      <div class="score-calc-row">
        <div class="score-calc-inputs">
          <div class="metric-box">
            <label>X_min (${Object.entries(exp.results).sort((a,b)=>a[1].ratio-b[1].ratio)[0][0].toUpperCase()})</label>
            <input type="text" readonly value="${exp.xMin.toFixed(4)}">
          </div>
          <div class="metric-box">
            <label>X_max (${Object.entries(exp.results).sort((a,b)=>b[1].ratio-a[1].ratio)[0][0].toUpperCase()})</label>
            <input type="text" readonly value="${exp.xMax.toFixed(4)}">
          </div>
          <div class="metric-box">
            <label>Spread (X_max − X_min)</label>
            <input type="text" readonly value="${exp.spread.toFixed(4)}">
          </div>
        </div>
        <div class="score-formula-display">
          Score = 1000 / ${exp.spread.toFixed(4)} = <strong>${exp.score.toFixed(2)}</strong>
        </div>
        <div class="score-final" style="color:${scoreColor(exp.score)}">${exp.score.toFixed(0)}</div>
      </div>
      <div style="margin-top:8px;font-size:.82rem;color:var(--txt-3);line-height:1.6">
        💡 ${exp.insight}
      </div>
    </div>

    <!-- Analysis (Findings & Conclusions) -->
    <div class="exp-analysis">
      <div class="analysis-grid">
        <div class="analysis-box findings">
          <div class="analysis-title">📋 Key Findings</div>
          <ul class="analysis-list">
            ${exp.findings.map(f => `<li>${f}</li>`).join('')}
          </ul>
        </div>
        <div class="analysis-box conclusion">
          <div class="analysis-title">🎯 Conclusion</div>
          <ul class="analysis-list">
            ${exp.conclusions.map(c => `<li>${c}</li>`).join('')}
          </ul>
        </div>
      </div>
    </div>

    <div class="exp-footer">
      <div style="font-size:.82rem;color:var(--txt-3)">
        Model: <code>${exp.modelFile}</code>
      </div>
      <a href="${ASSET_BASE}/models/${exp.modelFile}" download="${exp.modelFile}" class="btn btn-primary btn-sm">
        ⬇ Download Tokenizer JSON
      </a>
    </div>
  </article>
  `).join('');

  // Draw per-experiment charts after DOM is ready
  experiments.forEach(exp => createExpChart(exp));
}

function createExpChart(exp) {
  const ctx = document.getElementById(`chart-${exp.id}`);
  if (!ctx) return;
  const labels = LANGS.map(l => LANG_META[l].name);
  const ratios = LANGS.map(l => exp.results[l].ratio);
  const bgColors = LANGS.map(l => LANG_META[l].soft);
  const bdColors = LANGS.map(l => LANG_META[l].color);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Fertility Ratio',
        data: ratios,
        backgroundColor: bgColors,
        borderColor: bdColors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Ratio: ${ctx.raw.toFixed(4)} · Score contribution`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 } } },
        y: {
          min: 0, max: Math.ceil(exp.xMax) + 0.5,
          grid: { color: 'rgba(0,0,0,.05)' },
          ticks: { font: { family: 'JetBrains Mono', size: 10 } },
          title: { display: true, text: 'X (tokens/word)', font: { family: 'Inter', size: 10 } },
        },
      },
    },
  });
}
