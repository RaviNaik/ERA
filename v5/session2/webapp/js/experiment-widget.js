// js/experiment-widget.js — Renders one card per experiment
import { LANG_META, ASSET_BASE } from './data.js';

const LANGS = ['en', 'hi', 'te', 'kn'];
const MAX_RATIO = 2.5; // for ratio bar scaling

function scoreColor(score) {
  if (score >= 20000) return '#1db954';
  if (score >= 5000)  return '#5eba80';
  if (score >= 2000)  return '#5080d0';
  if (score >= 1000)  return '#d4902a';
  return '#e07c8c';
}

function rankEmoji(rank) {
  return ['\ud83e\udd47','\ud83e\udd48','\ud83e\udd49','4\ufe0f\u20e3','5\ufe0f\u20e3','6\ufe0f\u20e3','7\ufe0f\u20e3'][rank - 1] ?? rank;
}

// results.ratio is always tokens / faithfulUnits now
function ratioRow(lang, data) {
  const m = LANG_META[lang];
  if (!data) return '';
  const pct = Math.min((data.ratio / MAX_RATIO) * 100, 100).toFixed(1);
  const threshold = data.ratio <= 1.2
    ? `<span style="color:#5eba80;font-size:.75rem">\u2713&nbsp;&lt;&nbsp;1.2</span>`
    : `<span style="color:#e07c8c;font-size:.75rem">\u2717&nbsp;over&nbsp;1.2</span>`;
  const denomLabel = data.faithfulUnits
    ? `${data.faithfulUnits.toLocaleString()} units`
    : `${data.tokens.toLocaleString()} toks`;

  return `
  <tr>
    <td>
      <span class="lang-pill" style="background:${m.light};color:${m.color}">
        ${m.name}
      </span>
    </td>
    <td class="num">${data.tokens.toLocaleString()}</td>
    <td class="num">${(data.faithfulUnits || m.faithfulUnits).toLocaleString()}</td>
    <td>
      <div class="ratio-val">
        <span style="color:${m.color};font-weight:700">${data.ratio.toFixed(4)}</span>
        ${threshold}
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
    <div class="card-label" style="margin-bottom:8px">Per-Language Tokens Trained (budget: 2,500)</div>
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
        ${exp.subtitle ? `<div class="exp-subtitle" style="color:${exp.accent};font-size:.85rem;font-weight:600;margin-top:2px;margin-bottom:4px">${exp.subtitle}</div>` : ''}
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
        <div class="exp-results-title">Fertility Ratios (tokens \u00f7 faithful units)</div>
        <table class="ratio-table">
          <thead>
            <tr>
              <th>Language</th>
              <th>Tokens</th>
              <th>Faithful Units</th>
              <th>Ratio (X) &nbsp;&nbsp; Threshold</th>
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
            <label>Spread (X_max \u2212 X_min)</label>
            <input type="text" readonly value="${exp.spread.toFixed(4)}">
          </div>
        </div>
        <div class="score-formula-display">
          Score = 1000 / ${exp.spread.toFixed(4)} = <strong>${exp.score.toFixed(2)}</strong>
        </div>
        <div class="score-final" style="color:${scoreColor(exp.score)}">${exp.score.toFixed(0)}</div>
      </div>
      <div style="margin-top:8px;font-size:.82rem;color:var(--txt-3);line-height:1.6">
        \ud83d\udca1 ${exp.insight}
      </div>
    </div>

    <!-- Analysis (Findings & Conclusions) -->
    <div class="exp-analysis">
      <div class="analysis-grid">
        <div class="analysis-box findings">
          <div class="analysis-title">\ud83d\udccb Key Findings</div>
          <ul class="analysis-list">
            ${exp.findings.map(f => `<li>${f}</li>`).join('')}
          </ul>
        </div>
        <div class="analysis-box conclusion">
          <div class="analysis-title">\ud83c\udfaf Conclusion</div>
          <ul class="analysis-list">
            ${exp.conclusions.map(c => `<li>${c}</li>`).join('')}
          </ul>
        </div>
      </div>
    </div>

    <div class="exp-footer">
      <div style="font-size:.82rem;color:var(--txt-3)">
        Tokenizer: <code>${exp.modelFile}</code>
        &nbsp;&middot;&nbsp;
        Metrics: <code>${exp.metricsFile}</code>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <a href="${ASSET_BASE}/models/${exp.modelFile}" download="${exp.modelFile}" class="btn btn-primary btn-sm">
          \u2b07 Tokenizer JSON
        </a>
        <a href="${ASSET_BASE}/metrics/${exp.metricsFile}" download="${exp.metricsFile}" class="btn btn-outline btn-sm">
          \u2b07 Metrics JSON
        </a>
      </div>
    </div>
  </article>
  `).join('');

  // Draw per-experiment charts after DOM is ready
  experiments.forEach(exp => createExpChart(exp));
}

function createExpChart(exp) {
  const ctx = document.getElementById(`chart-${exp.id}`);
  if (!ctx) return;
  const labels   = LANGS.map(l => LANG_META[l].name);
  const ratios   = LANGS.map(l => exp.results[l]?.ratio ?? 0);
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
        annotation: {
          annotations: {
            threshold: {
              type: 'line', yMin: 1.2, yMax: 1.2,
              borderColor: 'rgba(224,124,140,0.6)', borderWidth: 1.5,
              borderDash: [4, 4],
              label: { content: '1.2 threshold', display: true, position: 'end', font: { size: 10 } }
            }
          }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` Ratio: ${ctx.raw.toFixed(4)} (tokens/faithful-unit)`
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 } } },
        y: {
          min: 0, max: Math.max(Math.ceil(exp.xMax * 10) / 10 + 0.1, 1.4),
          grid: { color: 'rgba(0,0,0,.05)' },
          ticks: { font: { family: 'JetBrains Mono', size: 10 } },
          title: { display: true, text: 'X (tokens / faithful-unit)', font: { family: 'Inter', size: 10 } },
        },
      },
    },
  });
}
