// ── app.js — main controller ──────────────────────────────────────

// ── Stage navigation ──────────────────────────────────────────────
let currentStage = 1;

function showStage(n) {
  document.querySelectorAll('.stage').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.stage-btn').forEach(b => b.classList.remove('active'));
  const stage = document.getElementById('stage-' + n);
  const btn   = document.getElementById('nav-btn-' + n);
  if (stage) stage.classList.remove('hidden');
  if (btn)   btn.classList.add('active');
  currentStage = n;

  // Init stage on first show
  if (n === 1 && !stage._init) { initStage1(); stage._init = true; }
  if (n === 2 && !stage._init) { initStage2(); stage._init = true; }
  if (n === 3 && !stage._init) { initStage3(); stage._init = true; }
  if (n === 4 && !stage._init) { initStage4(); stage._init = true; }
  if (n === 5 && !stage._init) { initStage5(); stage._init = true; }

  // Redraw charts on resize after show
  setTimeout(() => {
    if (n === 1) drawBenchChart('bench-chart', BENCH[currentBenchTab], BENCH_LABELS[currentBenchTab]);
    if (n === 3) { drawFertilityChart('fertility-chart'); drawVocabDonut('vocab-donut'); }
    if (n === 4) drawRetentionChart('retention-chart');
  }, 60);
}

// ── Generic tab switcher helper ───────────────────────────────────
function bindTabs(rowId, cb) {
  document.getElementById(rowId).querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById(rowId).querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      cb(btn.dataset.tab);
    });
  });
}

// ── Stage initialisers ────────────────────────────────────────────
let currentBenchTab = 'general';

function initStage1() {
  renderBenchTab('general');
  bindTabs('bench-tabs', key => {
    currentBenchTab = key;
    renderBenchTab(key);
  });
}

function initStage2() {
  renderDataTab('tier');
  bindTabs('data-tabs', key => renderDataTab(key));
}

function initStage3() {
  renderLangTable();
  setTimeout(() => {
    drawFertilityChart('fertility-chart');
    drawVocabDonut('vocab-donut');
    renderVocabLegend();
  }, 60);
}

function renderVocabLegend() {
  const top6 = VOCAB_ALLOC.slice(0, 6);
  // Just draw inline in donut card — already visible from donut
}

function initStage4() {
  renderPipeline(0);
  setTimeout(() => drawRetentionChart('retention-chart'), 60);
}

function initStage5() {
  renderPostTab('sft');
  bindTabs('post-tabs', key => renderPostTab(key));
}

// ── Boot ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Stage nav
  document.querySelectorAll('.stage-btn').forEach(btn => {
    btn.addEventListener('click', () => showStage(+btn.dataset.stage));
  });

  // Show stage 1 immediately
  showStage(1);

  // Redraw on window resize
  let resizeT;
  window.addEventListener('resize', () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => showStage(currentStage), 150);
  });

  // Animate KPI cards in
  document.querySelectorAll('.kpi-card').forEach((c, i) => {
    c.style.opacity = '0';
    c.style.transform = 'translateY(12px)';
    setTimeout(() => {
      c.style.transition = 'all 0.4s ease';
      c.style.opacity = '1';
      c.style.transform = 'translateY(0)';
    }, 120 + i * 80);
  });
});
