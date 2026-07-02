/* ===== WIDGET 1: Activations Exist for a Reason ===== */

const W1 = (() => {
  // updateProgress and NetworkViz are defined globally

  // ── Network layer definitions ─────────────────────────────────────────────
  const LAYERS_LINEAR = [
    { nodes: 2, label: 'Input',  type: 'input',  activation: null,      nodeLabels: ['x₁','x₂'] },
    { nodes: 1, label: 'Output', type: 'output', activation: 'sigmoid', nodeLabels: ['ŷ'] },
  ];
  const LAYERS_RELU = [
    { nodes: 2,  label: 'Input',   type: 'input',  activation: null,      nodeLabels: ['x₁','x₂'] },
    { nodes: 16, label: 'Hidden 1',type: 'hidden', activation: 'relu' },
    { nodes: 8,  label: 'Hidden 2',type: 'hidden', activation: 'relu' },
    { nodes: 1,  label: 'Output',  type: 'output', activation: 'sigmoid', nodeLabels: ['ŷ'] },
  ];

  // ── Data generation ──────────────────────────────────────────────────────
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

  let data = generateRings(300);
  let modelLinear = null, modelRelu = null;
  let accLinear = 0, accRelu = 0;

  // ── Model factories ───────────────────────────────────────────────────────
  function buildLinearModel() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 1, inputShape: [2], activation: 'sigmoid' }));
    m.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  function buildReluModel() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 16, inputShape: [2], activation: 'relu' }));
    m.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  // ── Decision boundary canvas ──────────────────────────────────────────────
  async function drawDecisionBoundary(canvas, model, points, labels, title, accent) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const minX = -6.5, maxX = 6.5, minY = -6.5, maxY = 6.5;

    const toPixX = x => ((x - minX) / (maxX - minX)) * W;
    const toPixY = y => H - ((y - minY) / (maxY - minY)) * H;

    if (model) {
      // Draw decision boundary grid
      const step = 0.18, gridX = [], gridY = [];
      for (let xi = minX; xi <= maxX; xi += step)
        for (let yi = minY; yi <= maxY; yi += step) {
          gridX.push(xi); gridY.push(yi);
        }

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
          if (p > 0.5) {
            ctx.fillStyle = `rgba(107,174,214,${0.2 + p * 0.25})`;
          } else {
            ctx.fillStyle = `rgba(224,123,106,${0.2 + (1 - p) * 0.25})`;
          }
          ctx.fillRect(px, py - ph, pw, ph);
        }
      }

      // Decision boundary line (0.5 contour) - draw as overlay dots
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

    // Draw data points
    for (let i = 0; i < points.length; i++) {
      const [xi, yi] = points[i];
      const cls = labels[i];
      ctx.beginPath();
      ctx.arc(toPixX(xi), toPixY(yi), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = cls === 0 ? 'rgba(224,123,106,0.85)' : 'rgba(107,174,214,0.85)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(100,90,120,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(toPixX(0), 0); ctx.lineTo(toPixX(0), H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, toPixY(0)); ctx.lineTo(W, toPixY(0)); ctx.stroke();
  }

  // ── Training ──────────────────────────────────────────────────────────────
  async function train() {
    const btn = document.getElementById('w1-train-btn');
    btn.disabled = true;
    btn.classList.add('loading');

    document.getElementById('w1-status').textContent = 'Generating data…';

    data = generateRings(300);
    const xs = tf.tensor2d(data.X);
    const ys = tf.tensor2d(data.y, [data.y.length, 1]);

    // Draw initial data
    const c1 = document.getElementById('w1-canvas-linear');
    const c2 = document.getElementById('w1-canvas-relu');
    await drawDecisionBoundary(c1, null, data.X, data.y, 'Linear', '#9b82d0');
    await drawDecisionBoundary(c2, null, data.X, data.y, 'ReLU', '#52b788');

    document.getElementById('w1-status').textContent = 'Training linear model…';
    updateProgress('w1-progress', 0);

    modelLinear = buildLinearModel();
    await modelLinear.fit(xs, ys, {
      epochs: 200,
      batchSize: 64,
      shuffle: true,
      callbacks: {
        onEpochEnd: (ep, logs) => {
          updateProgress('w1-progress', (ep + 1) / 200 * 50);
          document.getElementById('w1-status').textContent =
            `Linear training: epoch ${ep + 1}/200, loss ${logs.loss.toFixed(4)}`;
        }
      }
    });

    document.getElementById('w1-status').textContent = 'Training ReLU model…';
    modelRelu = buildReluModel();
    await modelRelu.fit(xs, ys, {
      epochs: 300,
      batchSize: 64,
      shuffle: true,
      callbacks: {
        onEpochEnd: (ep, logs) => {
          updateProgress('w1-progress', 50 + (ep + 1) / 300 * 50);
          document.getElementById('w1-status').textContent =
            `ReLU training: epoch ${ep + 1}/300, loss ${logs.loss.toFixed(4)}`;
        }
      }
    });

    // Compute accuracies
    const predLinear = modelLinear.predict(xs);
    const predRelu = modelRelu.predict(xs);
    const pl = await predLinear.data();
    const pr = await predRelu.data();
    predLinear.dispose(); predRelu.dispose();

    let correctL = 0, correctR = 0;
    data.y.forEach((gt, i) => {
      if ((pl[i] >= 0.5 ? 1 : 0) === gt) correctL++;
      if ((pr[i] >= 0.5 ? 1 : 0) === gt) correctR++;
    });
    accLinear = (correctL / data.y.length * 100).toFixed(1);
    accRelu = (correctR / data.y.length * 100).toFixed(1);

    document.getElementById('w1-acc-linear').textContent = accLinear + '%';
    document.getElementById('w1-acc-relu').textContent = accRelu + '%';

    xs.dispose(); ys.dispose();

    document.getElementById('w1-status').textContent = 'Done! Drawing boundaries…';
    await drawDecisionBoundary(c1, modelLinear, data.X, data.y, 'Linear', '#9b82d0');
    await drawDecisionBoundary(c2, modelRelu, data.X, data.y, 'ReLU', '#52b788');

    // Draw live network weights
    const wLinear = await NetworkViz.extractWeights(modelLinear);
    const wRelu   = await NetworkViz.extractWeights(modelRelu);
    NetworkViz.draw(document.getElementById('w1-net-linear'), LAYERS_LINEAR, wLinear);
    NetworkViz.draw(document.getElementById('w1-net-relu'),   LAYERS_RELU,   wRelu);

    document.getElementById('w1-status').textContent = `✓ Training complete`;
    updateProgress('w1-progress', 100);

    btn.disabled = false;
    btn.classList.remove('loading');
  }

  return { init: () => {
    // Draw skeleton network diagrams immediately
    NetworkViz.draw(document.getElementById('w1-net-linear'), LAYERS_LINEAR, null);
    NetworkViz.draw(document.getElementById('w1-net-relu'),   LAYERS_RELU,   null);

    // Draw initial blank canvases with data points after a brief delay
    setTimeout(async () => {
      data = generateRings(300);
      await drawDecisionBoundary(document.getElementById('w1-canvas-linear'), null, data.X, data.y, '', '#9b82d0');
      await drawDecisionBoundary(document.getElementById('w1-canvas-relu'), null, data.X, data.y, '', '#52b788');
    }, 300);

    document.getElementById('w1-train-btn').addEventListener('click', train);
  }};
})();
