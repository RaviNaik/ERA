// app.js — Main entry point; imports and initialises all widgets
import { EXPERIMENTS, LEADERBOARD } from './js/data.js';
import { renderHeroScores, renderLangCards, renderCorpusChart, initCorpusViewer } from './js/corpus-widget.js';
import { renderExperiments } from './js/experiment-widget.js';
import { renderComparison } from './js/comparison-widget.js';
import { renderScoreCalculator } from './js/score-calculator.js';

document.addEventListener('DOMContentLoaded', () => {

  // ── Corpus section ──────────────────────────────────────────
  renderHeroScores(EXPERIMENTS);
  renderLangCards();
  renderCorpusChart();
  initCorpusViewer();

  // ── Experiment cards ────────────────────────────────────────
  renderExperiments(EXPERIMENTS);

  // ── Comparison section ──────────────────────────────────────
  renderComparison(EXPERIMENTS);

  // ── Score calculator (pre-fill with best model's ratios) ───
  const best = LEADERBOARD[0];
  const defaultRatios = Object.fromEntries(
    Object.entries(best.results).map(([lang, r]) => [lang, r.ratio])
  );
  renderScoreCalculator(defaultRatios);

  // ── Smooth scroll for nav ───────────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Navbar active link highlight on scroll ──────────────────
  const sections = document.querySelectorAll('section[id], article[id]');
  const navLinks  = document.querySelectorAll('.nav-links a');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(l => {
          l.style.background = '';
          l.style.color = '';
        });
        const active = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (active) {
          active.style.background = 'rgba(96,85,216,0.12)';
          active.style.color = 'var(--accent)';
        }
      }
    });
  }, { rootMargin: '-40% 0px -50% 0px' });

  sections.forEach(s => observer.observe(s));
});
