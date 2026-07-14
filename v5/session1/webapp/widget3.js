/* ===== WIDGET 3: Embeddings Learn Similarity from Next-Token ===== */

const W3 = (() => {
  // ── Toy dataset ───────────────────────────────────────────────────────────
  const CHAINS = [
    ['cat',   'dog',   'cow'],
    ['apple', 'mango', 'banana'],
    ['eat',   'chase', 'see'],
  ];

  const ALL_TOKENS = CHAINS.flat();          // 9 tokens
  const VOCAB_SIZE = ALL_TOKENS.length;      // 9
  const TOKEN2ID   = {};
  ALL_TOKENS.forEach((t, i) => { TOKEN2ID[t] = i; });

  const CHAIN_META = [
    { label: 'cat → dog → cow',         color: '#e07b6a', emoji: '🐾' },
    { label: 'apple → mango → banana',  color: '#6baed6', emoji: '🍎' },
    { label: 'eat → chase → see',        color: '#52b788', emoji: '🏃' },
  ];
  const TOKEN_CHAIN = {};
  CHAINS.forEach((chain, ci) => chain.forEach(t => { TOKEN_CHAIN[t] = ci; }));

  // 6 unique bigrams (input → target)
  const BIGRAMS = [];
  for (const chain of CHAINS)
    for (let i = 0; i < chain.length - 1; i++)
      BIGRAMS.push([TOKEN2ID[chain[i]], TOKEN2ID[chain[i + 1]]]);

  // ── State ─────────────────────────────────────────────────────────────────
  let currentModel = null;
  let stopFlag     = false;

  // ── Read UI hyperparams ───────────────────────────────────────────────────
  function getHP() {
    const ep = parseInt(document.getElementById('w3-epochs-input').value, 10);
    const lr = parseFloat(document.getElementById('w3-lr-input').value);
    return {
      epochs: (!isNaN(ep) && ep >= 10) ? Math.min(ep, 1000) : 150,
      lr:     (!isNaN(lr) && lr >  0)  ? Math.min(lr, 1.0)  : 0.1,
    };
  }

  // ── Build model using Dense layer (mathematically identical to Embedding) ──
  function buildModel(lr) {
    const model = tf.sequential();
    // A Dense layer without bias on a one-hot input is identical to an Embedding lookup
    model.add(tf.layers.dense({
      units: 4,
      inputShape: [VOCAB_SIZE],
      useBias: false,
      name: 'emb',
    }));
    // No flatten needed because output is [batch, 4] (not [batch, 1, 4])
    model.add(tf.layers.dense({ units: VOCAB_SIZE, activation: 'softmax', name: 'out' }));
    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  }

  // ── Network Architecture Diagram ─────────────────────────────────────────
  function drawNetwork(canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f0d18';
    ctx.fillRect(0, 0, W, H);

    const LAYERS = [
      { label: 'Input\n(token id)',  n: VOCAB_SIZE, color: '#9b82d0', r: 9,  x: W * 0.12 },
      { label: 'Embedding\n(dim 4)', n: 4,          color: '#f0a04a', r: 14, x: W * 0.50 },
      { label: 'Softmax\n(vocab 9)', n: VOCAB_SIZE, color: '#52b788', r: 9,  x: W * 0.88 },
    ];

    // Precompute node positions
    const pos = LAYERS.map(({ n, x }) => {
      const step = (H - 60) / (n + 1);
      return Array.from({ length: n }, (_, i) => ({ x, y: 30 + step * (i + 1) }));
    });

    // Ghost connections (all-to-all between adjacent layers)
    for (let li = 0; li < 2; li++) {
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 0.7;
      for (const f of pos[li])
        for (const t of pos[li + 1]) {
          ctx.beginPath(); ctx.moveTo(f.x, f.y); ctx.lineTo(t.x, t.y); ctx.stroke();
        }
    }

    // Highlighted connections: one representative token per chain
    [0, 3, 6].forEach((tokIdx, ci) => {
      const color = CHAIN_META[ci].color;
      const from  = pos[0][tokIdx];
      ctx.lineWidth = 1.8;
      for (const to of pos[1]) {
        ctx.strokeStyle = color + '60';
        ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      }
    });

    // Draw nodes + labels
    LAYERS.forEach(({ r, color }, li) => {
      pos[li].forEach((p, ni) => {
        const isHL = li === 0 && [0, 3, 6].includes(ni);
        const ci   = li === 0 ? TOKEN_CHAIN[ALL_TOKENS[ni]] : null;
        const nc   = (isHL && ci !== null) ? CHAIN_META[ci].color : color;

        ctx.shadowColor = nc; ctx.shadowBlur = isHL ? 18 : 5;
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = nc + (isHL ? '' : '99'); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1; ctx.stroke();

        ctx.textBaseline = 'middle'; ctx.fillStyle = '#ffffff';
        if (li === 0) {
          ctx.font = 'bold 8px Inter,sans-serif';
          ctx.textAlign = 'right';
          ctx.fillText(ALL_TOKENS[ni], p.x - r - 3, p.y);
        } else if (li === 2) {
          ctx.font = 'bold 8px Inter,sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(ALL_TOKENS[ni], p.x + r + 3, p.y);
        } else {
          ctx.font = 'bold 9px Inter,sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`e${ni + 1}`, p.x, p.y);
        }
      });
    });

    // Layer title labels at top
    LAYERS.forEach(({ label, color, x }) => {
      ctx.font = '10px Inter,sans-serif'; ctx.fillStyle = color;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      label.split('\n').forEach((line, j) => ctx.fillText(line, x, 3 + j * 12));
    });
  }

  // ── Loss sparkline ────────────────────────────────────────────────────────
  function drawSparkline(losses, canvas) {
    if (!canvas || losses.length < 2) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const maxL = losses[0], minL = Math.min(...losses);
    const range = (maxL - minL) || 1e-6;
    const toY = l => H - 3 - ((l - minL) / range) * (H - 6);
    const toX = i => (losses.length < 2) ? 0 : (i / (losses.length - 1)) * W;

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#52b78860'); grad.addColorStop(1, '#52b78800');
    ctx.beginPath();
    losses.forEach((l, i) => i === 0 ? ctx.moveTo(toX(i), toY(l)) : ctx.lineTo(toX(i), toY(l)));
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    losses.forEach((l, i) => i === 0 ? ctx.moveTo(toX(i), toY(l)) : ctx.lineTo(toX(i), toY(l)));
    ctx.strokeStyle = '#52b788'; ctx.lineWidth = 2; ctx.stroke();
  }

  // ── PCA 2D ────────────────────────────────────────────────────────────────
  function pca2D(matrix) {
    const n = matrix.length, d = matrix[0].length;
    const mean = Array(d).fill(0);
    matrix.forEach(row => row.forEach((v, j) => { mean[j] += v / n; }));
    const C = matrix.map(row => row.map((v, j) => v - mean[j]));

    function powerIter(data, iters = 100) {
      let v = Array.from({ length: d }, (_, i) => (i % 2 === 0 ? 1 : -1) / Math.sqrt(d));
      for (let k = 0; k < iters; k++) {
        const Xv = data.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
        const nv  = Array(d).fill(0);
        data.forEach((row, i) => row.forEach((x, j) => { nv[j] += x * Xv[i] / n; }));
        const nm  = Math.sqrt(nv.reduce((s, x) => s + x * x, 0)) || 1e-12;
        v = nv.map(x => x / nm);
      }
      return v;
    }

    const pc1 = powerIter(C);
    const s1  = C.map(row => row.reduce((s, x, j) => s + x * pc1[j], 0));
    const D   = C.map((row, i) => row.map((x, j) => x - s1[i] * pc1[j]));
    const pc2 = powerIter(D);
    const s2  = D.map(row => row.reduce((s, x, j) => s + x * pc2[j], 0));
    return s1.map((v, i) => [v, s2[i]]);
  }

  // ── Embedding scatter plot ────────────────────────────────────────────────
  function drawPlot(canvas, coords) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f0d18'; ctx.fillRect(0, 0, W, H);

    const PAD = 54;
    const xs = coords.map(c => c[0]), ys = coords.map(c => c[1]);
    let [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
    let [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
    const rx = (maxX - minX) * 0.3 || 0.5, ry = (maxY - minY) * 0.3 || 0.5;
    minX -= rx; maxX += rx; minY -= ry; maxY += ry;
    const toX = x => PAD + ((x - minX) / (maxX - minX)) * (W - PAD * 2);
    const toY = y => H - PAD - ((y - minY) / (maxY - minY)) * (H - PAD * 2);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 0.8;
    for (let g = 0; g <= 4; g++) {
      ctx.beginPath(); ctx.moveTo(PAD + g*(W-PAD*2)/4, PAD); ctx.lineTo(PAD + g*(W-PAD*2)/4, H-PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, PAD + g*(H-PAD*2)/4); ctx.lineTo(W-PAD, PAD + g*(H-PAD*2)/4); ctx.stroke();
    }

    // Cluster halos
    CHAINS.forEach((chain, ci) => {
      const color = CHAIN_META[ci].color;
      const pts = chain.map(t => [toX(coords[TOKEN2ID[t]][0]), toY(coords[TOKEN2ID[t]][1])]);
      const cx  = pts.reduce((s, p) => s + p[0], 0) / 3;
      const cy  = pts.reduce((s, p) => s + p[1], 0) / 3;
      const r   = pts.reduce((m, p) => Math.max(m, Math.hypot(p[0] - cx, p[1] - cy)), 30) + 28;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0, color + '2a'); grad.addColorStop(1, color + '00');
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fillStyle = grad; ctx.fill();
    });

    // Next-token arrows
    const drawArrow = (x1, y1, x2, y2, color) => {
      const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx, dy);
      if (len < 2) return;
      const ux = dx/len, uy = dy/len, NR = 17;
      const [sx, sy, ex, ey] = [x1+ux*NR, y1+uy*NR, x2-ux*NR, y2-uy*NR];
      ctx.save();
      ctx.strokeStyle = color + 'bb'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
      const a = Math.atan2(ey-sy, ex-sx);
      ctx.fillStyle = color + 'bb';
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 9*Math.cos(a-0.4), ey - 9*Math.sin(a-0.4));
      ctx.lineTo(ex - 9*Math.cos(a+0.4), ey - 9*Math.sin(a+0.4));
      ctx.closePath(); ctx.fill(); ctx.restore();
    };

    CHAINS.forEach((chain, ci) => {
      const color = CHAIN_META[ci].color;
      for (let i = 0; i < chain.length - 1; i++)
        drawArrow(toX(coords[TOKEN2ID[chain[i]]][0]),   toY(coords[TOKEN2ID[chain[i]]][1]),
                  toX(coords[TOKEN2ID[chain[i+1]]][0]), toY(coords[TOKEN2ID[chain[i+1]]][1]), color);
    });

    // Token nodes
    const NR = 17;
    ALL_TOKENS.forEach((token, idx) => {
      const px = toX(coords[idx][0]), py = toY(coords[idx][1]);
      const color = CHAIN_META[TOKEN_CHAIN[token]].color;
      ctx.shadowColor = color; ctx.shadowBlur = 20;
      ctx.beginPath(); ctx.arc(px, py, NR, 0, Math.PI*2); ctx.fillStyle = color; ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.font = 'bold 10px Inter,sans-serif'; ctx.fillStyle = '#fff';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(token, px, py);
    });

    // Axis labels
    ctx.font = '10px Inter,sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('PC 1  →', W/2, H-12);
    ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('PC 2  →', 0, 0); ctx.restore();
  }

  // ── Clear embedding plot ──────────────────────────────────────────────────
  function clearPlot(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0f0d18'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '13px Inter,sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Train the model to see the embedding projection', canvas.width/2, canvas.height/2);
  }

  // ── NN table ──────────────────────────────────────────────────────────────
  function buildNNTable(embMatrix) {
    const el = document.getElementById('w3-nn-table');
    if (!el) return;
    const cosine = (a, b) => {
      const dot = a.reduce((s, v, i) => s + v * b[i], 0);
      return dot / (Math.sqrt(a.reduce((s,v)=>s+v*v,0)) * Math.sqrt(b.reduce((s,v)=>s+v*v,0)) + 1e-9);
    };
    el.innerHTML = `<tr><th>Token</th><th>Chain</th><th>Nearest Neighbour</th><th>2nd Nearest</th></tr>` +
      ALL_TOKENS.map((token, i) => {
        const sims = ALL_TOKENS.map((t2, j) => ({ t: t2, ci: TOKEN_CHAIN[t2], s: cosine(embMatrix[i], embMatrix[j]) }))
                               .filter(d => d.t !== token).sort((a, b) => b.s - a.s);
        const ci  = TOKEN_CHAIN[token];
        const c   = CHAIN_META[ci].color;
        const tick = d => d.ci === ci ? ' <span style="color:#52b788;font-weight:700">✓</span>' : '';
        const badge = d => `<span class="token-badge"><span class="legend-dot" style="background:${CHAIN_META[d.ci].color}"></span>${d.t}${tick(d)}</span> <small style="color:#9b94b0">(${d.s.toFixed(3)})</small>`;
        return `<tr>
          <td><span class="token-badge"><span class="legend-dot" style="background:${c}"></span><strong>${token}</strong></span></td>
          <td style="color:${c}">${CHAIN_META[ci].emoji}</td>
          <td>${badge(sims[0])}</td>
          <td>${badge(sims[1])}</td>
        </tr>`;
      }).join('');
  }

  function resetNNTable() {
    const el = document.getElementById('w3-nn-table');
    if (el) el.innerHTML = `<tr><th>Token</th><th>Chain</th><th>Nearest Neighbour</th><th>2nd Nearest</th></tr>
      <tr><td colspan="4" style="color:#9b94b0;text-align:center;padding:24px;">Train the model to see nearest neighbours</td></tr>`;
  }

  // ── Set button states ─────────────────────────────────────────────────────
  function setButtons(training) {
    const trainBtn = document.getElementById('w3-train-btn');
    const stopBtn  = document.getElementById('w3-stop-btn');
    const resetBtn = document.getElementById('w3-reset-btn');
    if (trainBtn) { trainBtn.disabled = training; training ? trainBtn.classList.add('loading') : trainBtn.classList.remove('loading'); }
    if (stopBtn)  stopBtn.disabled  = !training;
    if (resetBtn) resetBtn.disabled = training;
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  function reset() {
    stopFlag = true;
    currentModel = null;

    document.getElementById('w3-status').textContent   = 'Weights reset. Click Train to start fresh.';
    const prog = document.getElementById('w3-progress');
    if (prog) prog.style.width = '0%';

    // Clear sparkline
    const lc = document.getElementById('w3-loss-canvas');
    if (lc) { const ctx = lc.getContext('2d'); ctx.clearRect(0, 0, lc.width, lc.height); }

    // Clear embedding plot
    clearPlot(document.getElementById('w3-canvas'));

    // Reset NN table
    resetNNTable();

    const conc = document.getElementById('w3-conclusion');
    if (conc) { conc.style.display = 'none'; conc.innerHTML = ''; }

    setButtons(false);
  }

  // ── Main training ─────────────────────────────────────────────────────────
  async function train() {
    stopFlag = false;
    setButtons(true);

    const { epochs, lr } = getHP();
    const setStatus   = msg => { const e = document.getElementById('w3-status');   if (e) e.textContent = msg; };
    const setProgress = pct => { const e = document.getElementById('w3-progress'); if (e) e.style.width  = pct + '%'; };

    const conc = document.getElementById('w3-conclusion');
    if (conc) { conc.style.display = 'none'; conc.innerHTML = ''; }

    try {
      setStatus(`Building model (lr=${lr}, epochs=${epochs}, batch=6)…`);
      setProgress(2);
      await new Promise(r => setTimeout(r, 20)); // yield to browser for UI paint

      // Full-batch tensors — only 6 rows
      // We use one-hot inputs to avoid all tf.layers.embedding bugs
      const xs = tf.tidy(() => tf.oneHot(tf.tensor1d(BIGRAMS.map(b => b[0]), 'int32'), VOCAB_SIZE));
      const ys = tf.tensor1d(BIGRAMS.map(b => b[1]), 'float32'); // float32 to prevent floor() errors in cross-entropy

      // Build model
      const model = buildModel(lr);
      currentModel = model;

      // Draw static network diagram
      const netCanvas = document.getElementById('w3-net-canvas');
      if (netCanvas) drawNetwork(netCanvas);

      setStatus(`Training… 0/${epochs}`);

      const losses      = [];
      const lossCanvas  = document.getElementById('w3-loss-canvas');
      // Chunk size: yield to browser every N epochs so UI stays responsive
      const CHUNK = Math.max(1, Math.floor(epochs / 80));

      let ep = 0;
      let finalLoss = 0, finalAcc = 0;
      while (ep < epochs && !stopFlag) {
        const end = Math.min(ep + CHUNK, epochs);
        if (stopFlag) break;

        const h = await model.fit(xs, ys, {
          epochs: end,
          initialEpoch: ep,
          batchSize: BIGRAMS.length,  // full batch → 1 dispatch per epoch
          shuffle: false,
          verbose: 0,
        });

        if (stopFlag) break;

        const loss = h.history.loss[h.history.loss.length - 1];
        const acc  = (h.history.acc ?? h.history.accuracy ?? [0])[h.history.loss.length - 1] ?? 0;
        finalLoss = loss;
        finalAcc = acc;
        losses.push(loss);
        ep = end;

        setProgress((ep / epochs) * 100);
        drawSparkline(losses, lossCanvas);
        setStatus(`Epoch ${ep}/${epochs} — loss: ${loss.toFixed(4)}, acc: ${(acc * 100).toFixed(0)}%`);
        await new Promise(r => setTimeout(r, 0)); // yield to browser
      }

      xs.dispose(); ys.dispose();

      if (stopFlag) {
        setStatus('⏹ Training stopped.');
        setProgress(0);
        setButtons(false);
        return;
      }

      // Extract & visualise embeddings
      const embW  = model.getLayer('emb').getWeights()[0];
      const embD  = await embW.data();
      const dim   = embW.shape[1];
      const embMx = Array.from({ length: VOCAB_SIZE }, (_, i) =>
        Array.from(embD.slice(i * dim, (i + 1) * dim)));
      embW.dispose();

      const coords = pca2D(embMx);
      drawPlot(document.getElementById('w3-canvas'), coords);
      buildNNTable(embMx);

      setStatus('✓ Done! Chain-mates cluster — learned only from next-token prediction.');
      setProgress(100);

      if (conc) {
        conc.style.display = 'block';
        conc.innerHTML = `<strong>Conclusion:</strong> The model reached a loss of <strong>${finalLoss.toFixed(4)}</strong> and accuracy of <strong>${(finalAcc*100).toFixed(0)}%</strong>. Without any explicit notion of similarity, tokens appearing in similar contexts naturally clustered together.`;
      }

    } catch (err) {
      console.error('[W3] Training error:', err);
      document.getElementById('w3-status').textContent = '❌ ' + err.message;
    }

    setButtons(false);
  }

  // ── Stop ──────────────────────────────────────────────────────────────────
  function stop() {
    stopFlag = true;
    if (currentModel) currentModel.stopTraining = true;
  }

  return {
    init() {
      document.getElementById('w3-train-btn').addEventListener('click', train);
      document.getElementById('w3-stop-btn').addEventListener('click', stop);
      document.getElementById('w3-reset-btn').addEventListener('click', reset);

      // Draw static network + placeholder on load
      const netCanvas = document.getElementById('w3-net-canvas');
      if (netCanvas) drawNetwork(netCanvas);
      clearPlot(document.getElementById('w3-canvas'));
      setButtons(false);
    }
  };
})();
