// ── charts.js — all Canvas drawing functions ──────────────────────

function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// Responsive DPI scaling
function scaleCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

// ── Grouped bar chart (benchmarks) ──────────────────────────────
function drawBenchChart(canvasId, rows, label) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = canvas.parentElement.clientWidth || 460;
  const H = 270;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {top:20,right:20,bottom:56,left:38};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top  - pad.bottom;

  const maxVal = 105;
  const n = rows.length;
  const groupW = cw / n;
  const bW = groupW * 0.32;
  const gap = groupW * 0.06;

  // Grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  [0,25,50,75,100].forEach(v => {
    const y = pad.top + ch - (v / maxVal * ch);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(v, pad.left - 4, y + 3);
  });

  rows.forEach((r, i) => {
    const x0 = pad.left + i * groupW + groupW * 0.1;

    // Gemma bar
    const yG = pad.top + ch - (r.gemma / maxVal * ch);
    const hG = r.gemma / maxVal * ch;
    ctx.fillStyle = 'rgba(107,174,214,0.75)';
    roundRect(ctx, x0, yG, bW, hG, 3);

    // Bharat bar
    const yB = pad.top + ch - (r.bharat / maxVal * ch);
    const hB = r.bharat / maxVal * ch;
    const bc = r.gap === '↑' ? 'rgba(82,183,136,0.82)' : r.gap === 'close' ? 'rgba(244,167,111,0.82)' : 'rgba(155,130,208,0.82)';
    ctx.fillStyle = bc;
    roundRect(ctx, x0 + bW + gap, yB, bW, hB, 3);

    // Value labels
    ctx.fillStyle = '#4a90c4';
    ctx.font = 'bold 9px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(r.gemma, x0 + bW/2, yG - 3);
    ctx.fillStyle = r.gap === '↑' ? '#3a9e72' : r.gap === 'close' ? '#d4824a' : '#7c5cbf';
    ctx.fillText(r.bharat, x0 + bW + gap + bW/2, yB - 3);

    // X label
    ctx.fillStyle = '#6b6480';
    ctx.font = '9.5px Inter';
    ctx.textAlign = 'center';
    const lx = x0 + bW + gap/2;
    wrapText(ctx, r.name, lx, pad.top + ch + 14, groupW * 0.85, 11);
  });

  // Legend
  ctx.fillStyle = 'rgba(107,174,214,0.82)';
  ctx.fillRect(pad.left, H - 12, 10, 9);
  ctx.fillStyle = '#5a5070';
  ctx.font = '10px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Gemma 4 31B', pad.left + 13, H - 4);
  ctx.fillStyle = 'rgba(82,183,136,0.82)';
  ctx.fillRect(pad.left + 100, H - 12, 10, 9);
  ctx.fillText('Bharat-40B', pad.left + 113, H - 4);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

function wrapText(ctx, text, x, y, maxW, lineH) {
  const words = text.split(' ');
  let line = '';
  let ly = y;
  for (let i = 0; i < words.length; i++) {
    const test = line + words[i] + ' ';
    if (ctx.measureText(test).width > maxW && i > 0) {
      ctx.fillText(line, x, ly);
      line = words[i] + ' ';
      ly += lineH;
    } else { line = test; }
  }
  ctx.fillText(line, x, ly);
}

// ── Horizontal bar chart (fertility) ────────────────────────────
function drawFertilityChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const show = LANGS.filter(l => l.prio <= 2);
  const W = canvas.parentElement.clientWidth || 440;
  const H = show.length * 26 + 50;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const labW = 82, pad = {top:30, right:16, bottom:14, left:labW + 8};
  const barW = W - pad.left - pad.right;
  const maxVal = 6;

  ctx.fillStyle = '#9b94b0';
  ctx.font = 'bold 10px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Tokens per word', W / 2, 14);

  show.forEach((lang, i) => {
    const y = pad.top + i * 26;
    // Label
    ctx.fillStyle = '#2d2640';
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(lang.lang, labW, y + 14);

    // Current bar (faint)
    const xW1 = (lang.curr / maxVal) * barW;
    ctx.fillStyle = 'rgba(224,123,106,0.25)';
    ctx.beginPath(); ctx.roundRect(pad.left, y + 4, xW1, 9, 3); ctx.fill();

    // Target bar
    const xW2 = (lang.target / maxVal) * barW;
    ctx.fillStyle = 'rgba(82,183,136,0.75)';
    ctx.beginPath(); ctx.roundRect(pad.left, y + 4, xW2, 9, 3); ctx.fill();

    // Values
    ctx.fillStyle = '#c85a47';
    ctx.font = '9px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(lang.curr.toFixed(1), pad.left + xW1 + 3, y + 12);

    ctx.fillStyle = '#3a9e72';
    ctx.font = 'bold 9px Inter';
    ctx.fillText(lang.target.toFixed(1), pad.left + xW2 + 3, y + 12);
  });
}

// ── Donut chart (vocab allocation) ──────────────────────────────
function drawVocabDonut(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const SIZE = 260;
  canvas.style.width  = SIZE + 'px';
  canvas.style.height = SIZE + 'px';
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = SIZE * dpr;
  canvas.height = SIZE * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const cx = SIZE/2, cy = SIZE/2, R = 100, r = 60;
  const total = VOCAB_ALLOC.reduce((a,v) => a + v.val, 0);
  let angle = -Math.PI/2;

  VOCAB_ALLOC.forEach(seg => {
    const slice = (seg.val / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = seg.color + 'cc';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    angle += slice;
  });

  // Hole
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.82)';
  ctx.fill();

  // Center text
  ctx.fillStyle = '#2d2640';
  ctx.font = 'bold 22px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('160K', cx, cy + 3);
  ctx.fillStyle = '#9b94b0';
  ctx.font = '10px Inter';
  ctx.fillText('tokens', cx, cy + 16);
}

// ── Domain mix donut (Stage 2) ───────────────────────────────────
function drawDomainDonut(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const SIZE = 260;
  canvas.style.width  = SIZE + 'px';
  canvas.style.height = SIZE + 'px';
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = SIZE * dpr;
  canvas.height = SIZE * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const cx = SIZE/2, cy = SIZE/2, R = 108, r = 65;
  const total = DOMAIN_MIX.reduce((a,v) => a + v.pct, 0);
  let angle = -Math.PI/2;

  DOMAIN_MIX.forEach(seg => {
    const slice = (seg.pct / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = seg.color + 'cc';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    angle += slice;
  });

  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.82)';
  ctx.fill();
  ctx.fillStyle = '#2d2640';
  ctx.font = 'bold 18px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('15–18T', cx, cy + 3);
  ctx.fillStyle = '#9b94b0';
  ctx.font = '10px Inter';
  ctx.fillText('tokens', cx, cy + 16);
}

// ── Retention funnel (Stage 4) ───────────────────────────────────
function drawRetentionChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = canvas.parentElement.clientWidth || 860;
  const H = 200;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {top:24,right:16,bottom:40,left:42};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;
  const n  = RETENTION.stages.length;
  const xStep = cw / (n - 1);
  const datasets = [
    {key:'english', color:'#6baed6', label:'English (D1)'},
    {key:'indic',   color:'#52b788', label:'Indic (Tier 2)'},
    {key:'code',    color:'#9b82d0', label:'Code (D3)'},
  ];

  // Grid
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  [0,25,50,75,100].forEach(v => {
    const y = pad.top + ch - (v/100 * ch);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(v+'%', pad.left - 4, y + 3);
  });

  // X labels
  RETENTION.stages.forEach((s, i) => {
    const x = pad.left + i * xStep;
    ctx.fillStyle = '#9b94b0';
    ctx.font = '8.5px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(s, x, H - 4);
  });

  // Lines
  datasets.forEach(ds => {
    const vals = RETENTION[ds.key];
    ctx.beginPath();
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = 2.5;
    vals.forEach((v, i) => {
      const x = pad.left + i * xStep;
      const y = pad.top + ch - (v/100 * ch);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots + values
    vals.forEach((v, i) => {
      const x = pad.left + i * xStep;
      const y = pad.top + ch - (v/100 * ch);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI*2);
      ctx.fillStyle = ds.color;
      ctx.fill();
      if (i === 0 || i === vals.length - 1) {
        ctx.fillStyle = ds.color;
        ctx.font = 'bold 9px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(v+'%', x, y - 6);
      }
    });
  });

  // Legend
  let lx = pad.left;
  datasets.forEach(ds => {
    ctx.fillStyle = ds.color;
    ctx.fillRect(lx, pad.top - 14, 14, 7);
    ctx.fillStyle = '#6b6480';
    ctx.font = '9.5px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(ds.label, lx + 17, pad.top - 8);
    lx += 110;
  });
}
