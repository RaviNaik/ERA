// js/comparison-widget.js — Cross-experiment comparison charts and leaderboard
import { LANG_META, LEADERBOARD } from './data.js';

const LANGS = ['en', 'hi', 'te', 'kn'];

export function renderComparison(experiments) {
  renderComparisonChart(experiments);
  renderScoreChart(experiments);
  renderLeaderboard();
}

function renderComparisonChart(experiments) {
  const ctx = document.getElementById('comparisonChart');
  if (!ctx) return;

  // One dataset per language, one bar group per experiment
  const datasets = LANGS.map(lang => {
    const m = LANG_META[lang];
    return {
      label: m.name,
      data: experiments.map(e => e.results[lang].ratio),
      backgroundColor: m.soft,
      borderColor: m.color,
      borderWidth: 2,
      borderRadius: 6,
    };
  });

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: experiments.map(e => e.step),
      datasets,
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, font: { family: 'Inter', size: 11 } } },
        tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.raw.toFixed(4)}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 } } },
        y: {
          grid: { color: 'rgba(0,0,0,.05)' },
          title: { display: true, text: 'Fertility Ratio (tokens/faithful-unit)', font: { family: 'Inter', size: 11 } },
          ticks: { font: { family: 'JetBrains Mono', size: 10 } },
        },
      },
    },
  });
}

function renderScoreChart(experiments) {
  const ctx = document.getElementById('scoreChart');
  if (!ctx) return;

  const scoreColorMap = s => s >= 1200 ? '#5eba80' : s >= 900 ? '#5080d0' : s >= 600 ? '#d4902a' : '#e07c8c';

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: experiments.map(e => e.step),
      datasets: [{
        label: 'Score',
        data: experiments.map(e => e.score),
        backgroundColor: experiments.map(e => scoreColorMap(e.score) + 'bb'),
        borderColor: experiments.map(e => scoreColorMap(e.score)),
        borderWidth: 2,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` Score: ${c.raw.toFixed(2)}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 } } },
        y: {
          grid: { color: 'rgba(0,0,0,.05)' },
          title: { display: true, text: '1000 / spread', font: { family: 'Inter', size: 11 } },
          ticks: { font: { family: 'JetBrains Mono', size: 10 } },
        },
      },
    },
  });
}

function renderLeaderboard() {
  const el = document.getElementById('leaderboard');
  if (!el) return;

  const maxScore = LEADERBOARD[0].score;

  el.innerHTML = `
  <table class="lb-table">
    <thead>
      <tr>
        <th>Rank</th>
        <th>Experiment</th>
        <th>Score</th>
        <th>Score Bar</th>
        <th>X_min</th>
        <th>X_max</th>
        <th>Spread</th>
        <th>Actual Vocab</th>
      </tr>
    </thead>
    <tbody>
      ${LEADERBOARD.map((exp, i) => {
        const pct = ((exp.score / maxScore) * 100).toFixed(1);
        const scoreCol = exp.score >= 1200 ? '#5eba80' : exp.score >= 900 ? '#5080d0' : exp.score >= 600 ? '#d4902a' : '#e07c8c';
        const rankE = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣'][i];
        return `
        <tr>
          <td class="lb-rank">${rankE}</td>
          <td>
            <a href="#${exp.id}" style="font-weight:600;color:${exp.accent}">${exp.step}</a>
            <div style="font-size:.78rem;color:var(--txt-3)">${exp.name}</div>
          </td>
          <td><span class="lb-score" style="color:${scoreCol}">${exp.score.toFixed(2)}</span></td>
          <td style="min-width:100px">
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:${pct}%;background:${scoreCol}"></div>
            </div>
          </td>
          <td class="num">${exp.xMin.toFixed(4)}</td>
          <td class="num">${exp.xMax.toFixed(4)}</td>
          <td class="num">${exp.spread.toFixed(4)}</td>
          <td class="num">${parseInt(exp.config['Actual Vocab'].replace(/,/g,'')).toLocaleString()}</td>
        </tr>`;
      }).join('')}
    </tbody>
  </table>`;
}
