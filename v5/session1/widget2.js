/* ===== WIDGET 2: Depth Without Nonlinearity is a Lie ===== */

const W2 = (() => {
  // ── Network layer definitions ─────────────────────────────────────────────
  const LAYERS_1L = [
    { nodes: 2, label: 'Input',  type: 'input',  activation: null,      nodeLabels: ['x₁','x₂'] },
    { nodes: 1, label: 'Output', type: 'output', activation: 'sigmoid', nodeLabels: ['ŷ'] },
  ];
  const LAYERS_5L_LINEAR = [
    { nodes: 2,  label: 'Input',   type: 'input',  activation: null },
    { nodes: 16, label: 'Layer 1', type: 'hidden', activation: 'linear' },
    { nodes: 16, label: 'Layer 2', type: 'hidden', activation: 'linear' },
    { nodes: 16, label: 'Layer 3', type: 'hidden', activation: 'linear' },
    { nodes: 16, label: 'Layer 4', type: 'hidden', activation: 'linear' },
    { nodes: 1,  label: 'Output',  type: 'output', activation: 'sigmoid' },
  ];
  const LAYERS_5L_RELU = [
    { nodes: 2,  label: 'Input',   type: 'input',  activation: null },
    { nodes: 16, label: 'Layer 1', type: 'hidden', activation: 'relu' },
    { nodes: 16, label: 'Layer 2', type: 'hidden', activation: 'relu' },
    { nodes: 16, label: 'Layer 3', type: 'hidden', activation: 'relu' },
    { nodes: 16, label: 'Layer 4', type: 'hidden', activation: 'relu' },
    { nodes: 1,  label: 'Output',  type: 'output', activation: 'sigmoid' },
  ];

  let data = null;

  function generateRings(n = 300) {
    const X = [], y = [];
    const inner = Math.floor(n / 2), outer = n - inner;
    for (let i = 0; i < inner; i++) {
      const r = 1.5 + Math.random() * 1.0;
      const theta = Math.random() * 2 * Math.PI;
      X.push([r * Math.cos(theta) + (Math.random() - 0.5) * 0.5,
              r * Math.sin(theta) + (Math.random() - 0.5) * 0.5]);
      y.push(0);
    }
    for (let i = 0; i < outer; i++) {
      const r = 3.5 + Math.random() * 1.2;
      const theta = Math.random() * 2 * Math.PI;
      X.push([r * Math.cos(theta) + (Math.random() - 0.5) * 0.5,
              r * Math.sin(theta) + (Math.random() - 0.5) * 0.5]);
      y.push(1);
    }
    return { X, y };
  }

  // ── Model factories ───────────────────────────────────────────────────────
  function build1LayerLinear() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid',
      kernelInitializer: 'glorotUniform' }));
    m.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  function build5LayerLinear() {
    const m = tf.sequential();
    // 5 linear layers — NO activations; collapses to a single linear map
    m.add(tf.layers.dense({ units: 16, inputShape: [2], activation: 'linear', useBias: true }));
    m.add(tf.layers.dense({ units: 16, activation: 'linear', useBias: true }));
    m.add(tf.layers.dense({ units: 16, activation: 'linear', useBias: true }));
    m.add(tf.layers.dense({ units: 16, activation: 'linear', useBias: true }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid', useBias: true }));
    m.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  function build5LayerRelu() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 16, inputShape: [2], activation: 'relu' }));
    m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  // ── Decision boundary drawing (same as W1 but parameterised) ─────────────
  async function drawBoundary(canvasId, model, points, labels, accent) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const minX = -6.5, maxX = 6.5, minY = -6.5, maxY = 6.5;
    const toPixX = x => ((x - minX) / (maxX - minX)) * W;
    const toPixY = y => H - ((y - minY) / (maxY - minY)) * H;

    if (model) {
      const step = 0.18, gridX = [], gridY = [];
      for (let xi = minX; xi <= maxX; xi += step)
        for (let yi = minY; yi <= maxY; yi += step) { gridX.push(xi); gridY.push(yi); }

      const inputTensor = tf.tensor2d(gridX.map((xi, i) => [xi, gridY[i]]));
      const preds = await model.predict(inputTensor).data();
      inputTensor.dispose();

      let idx = 0;
      for (let xi = minX; xi <= maxX; xi += step) {
        for (let yi = minY; yi <= maxY; yi += step) {
          const p = preds[idx++];
          const px = toPixX(xi), py = toPixY(yi);
          const pw = Math.ceil((step / (maxX - minX)) * W) + 1;
          const ph = Math.ceil((step / (maxY - minY)) * H) + 1;
          ctx.fillStyle = p > 0.5
            ? `rgba(107,174,214,${0.2 + p * 0.25})`
            : `rgba(224,123,106,${0.2 + (1 - p) * 0.25})`;
          ctx.fillRect(px, py - ph, pw, ph);
        }
      }

      // Boundary line
      idx = 0;
      for (let xi = minX; xi <= maxX; xi += step) {
        for (let yi = minY; yi <= maxY; yi += step) {
          const p = preds[idx++];
          if (Math.abs(p - 0.5) < 0.045) {
            ctx.fillStyle = accent;
            ctx.beginPath();
            ctx.arc(toPixX(xi), toPixY(yi), 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    for (let i = 0; i < points.length; i++) {
      const [xi, yi] = points[i];
      ctx.beginPath();
      ctx.arc(toPixX(xi), toPixY(yi), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = labels[i] === 0 ? 'rgba(224,123,106,0.85)' : 'rgba(107,174,214,0.85)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(100,90,120,0.2)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(toPixX(0), 0); ctx.lineTo(toPixX(0), H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, toPixY(0)); ctx.lineTo(W, toPixY(0)); ctx.stroke();
  }

  // ── Compute weight matrix product for first 4 linear layers ──────────────
  async function computeAndShowMatrixProduct(model5Linear) {
    // Extract weight matrices of first 4 dense layers (all linear)
    // Each layer: kernel shape [in, units]
    const layers = model5Linear.layers;
    // layers[0]: dense(16, input=2), layers[1..3]: dense(16), each linear
    // Product: W_final = W4 @ W3 @ W2 @ W1  (shapes: 2→16→16→16→16)
    // We'll compute effective 2→16 map
    let W = tf.tensor(await layers[0].getWeights()[0].data(),
                       layers[0].getWeights()[0].shape);

    for (let l = 1; l <= 3; l++) {
      const wl = layers[l].getWeights()[0]; // [16,16]
      W = tf.matMul(W, wl);
    }

    const wData = await W.data();
    const shape = W.shape; // [2, 16]
    W.dispose();

    // Show as 2×16 matrix preview
    const el = document.getElementById('w2-matrix-product');
    if (!el) return;

    const rows = shape[0], cols = shape[1];
    let html = `<div style="font-size:11.5px;color:#6b6480;margin-bottom:8px;">
      Effective weight matrix <strong>W = W₄ · W₃ · W₂ · W₁</strong> (${rows}×${cols}) — 
      this is still a single linear projection:
    </div>`;

    for (let r = 0; r < rows; r++) {
      html += '<div class="matrix-row">';
      for (let c = 0; c < Math.min(cols, 8); c++) {
        html += `<span class="matrix-val">${wData[r * cols + c].toFixed(3)}</span>`;
      }
      if (cols > 8) html += `<span class="matrix-val" style="opacity:0.5;">…</span>`;
      html += '</div>';
    }
    html += `<div style="margin-top:8px;font-size:11.5px;color:#52b788;font-weight:600;">
      ✓ 5 weight matrices multiplied → one 2×${cols} matrix. No new expressiveness.
    </div>`;
    el.innerHTML = html;
  }

  // ── Accuracy helper ───────────────────────────────────────────────────────
  async function getAccuracy(model, xs) {
    const preds = await model.predict(xs).data();
    let correct = 0;
    data.y.forEach((gt, i) => { if ((preds[i] >= 0.5 ? 1 : 0) === gt) correct++; });
    return (correct / data.y.length * 100).toFixed(1);
  }

  // ── Main training ─────────────────────────────────────────────────────────
  async function train() {
    const btn = document.getElementById('w2-train-btn');
    btn.disabled = true;
    btn.classList.add('loading');

    data = generateRings(300);
    const xs = tf.tensor2d(data.X);
    const ys = tf.tensor2d(data.y, [data.y.length, 1]);

    // Clear canvases
    await drawBoundary('w2-canvas-1layer', null, data.X, data.y, '#6baed6');
    await drawBoundary('w2-canvas-5linear', null, data.X, data.y, '#6baed6');
    await drawBoundary('w2-canvas-5relu', null, data.X, data.y, '#52b788');

    const setStatus = (msg) => { document.getElementById('w2-status').textContent = msg; };
    const setProgress = (pct) => {
      const el = document.getElementById('w2-progress');
      if (el) el.style.width = pct + '%';
    };

    // ─── 1-layer linear ───────────────────────────
    setStatus('Training 1-layer linear…');
    const m1 = build1LayerLinear();
    await m1.fit(xs, ys, {
      epochs: 200, batchSize: 64, shuffle: true,
      callbacks: { onEpochEnd: (ep) => setProgress((ep + 1) / 200 * 33) }
    });
    const acc1 = await getAccuracy(m1, xs);
    document.getElementById('w2-acc-1layer').textContent = acc1 + '%';
    await drawBoundary('w2-canvas-1layer', m1, data.X, data.y, '#6baed6');
    const w1Mat = await NetworkViz.extractWeights(m1);
    NetworkViz.draw(document.getElementById('w2-net-1layer'), LAYERS_1L, w1Mat);

    // ─── 5-layer linear ───────────────────────────
    setStatus('Training 5-layer linear (no activations)…');
    const m5l = build5LayerLinear();
    await m5l.fit(xs, ys, {
      epochs: 200, batchSize: 64, shuffle: true,
      callbacks: { onEpochEnd: (ep) => setProgress(33 + (ep + 1) / 200 * 33) }
    });
    const acc5l = await getAccuracy(m5l, xs);
    document.getElementById('w2-acc-5linear').textContent = acc5l + '%';
    await drawBoundary('w2-canvas-5linear', m5l, data.X, data.y, '#6baed6');
    const w5lMat = await NetworkViz.extractWeights(m5l);
    NetworkViz.draw(document.getElementById('w2-net-5linear'), LAYERS_5L_LINEAR, w5lMat);

    // Show matrix product
    await computeAndShowMatrixProduct(m5l);

    // ─── 5-layer ReLU ─────────────────────────────
    setStatus('Training 5-layer ReLU…');
    const m5r = build5LayerRelu();
    await m5r.fit(xs, ys, {
      epochs: 300, batchSize: 64, shuffle: true,
      callbacks: { onEpochEnd: (ep) => setProgress(66 + (ep + 1) / 300 * 34) }
    });
    const acc5r = await getAccuracy(m5r, xs);
    document.getElementById('w2-acc-5relu').textContent = acc5r + '%';
    await drawBoundary('w2-canvas-5relu', m5r, data.X, data.y, '#52b788');
    const w5rMat = await NetworkViz.extractWeights(m5r);
    NetworkViz.draw(document.getElementById('w2-net-5relu'), LAYERS_5L_RELU, w5rMat);

    xs.dispose(); ys.dispose();
    setStatus('✓ Training complete');
    setProgress(100);
    btn.disabled = false;
    btn.classList.remove('loading');
  }

  return { init: () => {
    // Draw skeleton network diagrams immediately
    NetworkViz.draw(document.getElementById('w2-net-1layer'),   LAYERS_1L,        null);
    NetworkViz.draw(document.getElementById('w2-net-5linear'),  LAYERS_5L_LINEAR, null);
    NetworkViz.draw(document.getElementById('w2-net-5relu'),    LAYERS_5L_RELU,   null);

    setTimeout(async () => {
      data = generateRings(300);
      await drawBoundary('w2-canvas-1layer', null, data.X, data.y, '#6baed6');
      await drawBoundary('w2-canvas-5linear', null, data.X, data.y, '#6baed6');
      await drawBoundary('w2-canvas-5relu', null, data.X, data.y, '#52b788');
    }, 400);
    document.getElementById('w2-train-btn').addEventListener('click', train);
  }};
})();
