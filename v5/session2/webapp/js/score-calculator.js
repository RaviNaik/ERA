// js/score-calculator.js — Interactive score calculator widget
import { LANG_META } from './data.js';

const LANGS = ['en', 'hi', 'te', 'kn'];
const LANG_LABELS = { en: 'English (X1)', hi: 'Hindi (X2)', te: 'Telugu (X3)', kn: 'Kannada (X4)' };

export function renderScoreCalculator(defaultRatios) {
  renderCalcInputs(defaultRatios);
}

function renderCalcInputs(defaults) {
  const grid = document.getElementById('calc-grid');
  const result = document.getElementById('calc-result');
  if (!grid || !result) return;

  grid.innerHTML = LANGS.map(lang => {
    const m = LANG_META[lang];
    const def = defaults[lang]?.toFixed(4) ?? '1.5000';
    return `
    <div class="metric-box">
      <label style="color:${m.color}">${m.flag} ${LANG_LABELS[lang]}</label>
      <input type="number" id="calc-${lang}" step="0.01" min="1" max="10" value="${def}"
        style="border-color:${m.color}"
        oninput="window._recalcScore()">
    </div>`;
  }).join('');

  window._recalcScore = () => {
    const vals = LANGS.map(lang => {
      const v = parseFloat(document.getElementById(`calc-${lang}`)?.value);
      return isNaN(v) ? null : v;
    });

    if (vals.some(v => v === null)) {
      result.innerHTML = '<span style="color:var(--txt-3)">Enter valid ratios for all languages…</span>';
      return;
    }

    const xMin = Math.min(...vals);
    const xMax = Math.max(...vals);
    const spread = xMax - xMin;
    const score = spread > 0 ? (1000 / spread) : Infinity;
    const minLang = LANG_META[LANGS[vals.indexOf(xMin)]].name;
    const maxLang = LANG_META[LANGS[vals.indexOf(xMax)]].name;
    const scoreCol = score >= 1200 ? '#5eba80' : score >= 900 ? '#5080d0' : score >= 600 ? '#d4902a' : '#e07c8c';

    result.innerHTML = `
      <span class="calc-score-val" style="color:${scoreCol}">
        ${isFinite(score) ? score.toFixed(2) : '∞'}
      </span>
      <span>
        = 1000 / (${xMax.toFixed(4)} − ${xMin.toFixed(4)})
        &nbsp;·&nbsp; Spread: ${spread.toFixed(4)}
        &nbsp;·&nbsp; Best: ${minLang} (${xMin.toFixed(4)})
        &nbsp;·&nbsp; Worst: ${maxLang} (${xMax.toFixed(4)})
      </span>`;
  };

  // Initial calculation
  window._recalcScore();
}
