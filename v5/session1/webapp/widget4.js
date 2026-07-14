/* ===== WIDGET 4: Memorization vs Generalization ===== */

const W4 = (() => {
  let stopRequested = false;

  // ── Network layer definitions ─────────────────────────────────────────────
  const LAYERS_MODEL = [
    { nodes: 2,  label: 'Input',   type: 'input',  activation: null,      nodeLabels: ['x₁','x₂'] },
    { nodes: 64, label: 'Hidden 1',type: 'hidden', activation: 'relu' },
    { nodes: 64, label: 'Hidden 2',type: 'hidden', activation: 'relu' },
    { nodes: 32, label: 'Hidden 3',type: 'hidden', activation: 'relu' },
    { nodes: 1,  label: 'Output',  type: 'output', activation: 'sigmoid', nodeLabels: ['\u0177'] },
  ];

  // \u2500\u2500 Dataset generation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Noisy moons / spirals classification
  function generateDataset(n) {
    const X = [], y = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < half; i++) {
      const theta = (i / half) * Math.PI + (Math.random() - 0.5) * 0.6;
      const r = 1.0 + (Math.random() - 0.5) * 0.4;
      X.push([r * Math.cos(theta), r * Math.sin(theta) + (Math.random() - 0.5) * 0.5]);
      y.push(0);
    }
    for (let i = 0; i < n - half; i++) {
      const theta = (i / (n - half)) * Math.PI + Math.PI + (Math.random() - 0.5) * 0.6;
      const r = 1.0 + (Math.random() - 0.5) * 0.4;
      X.push([r * Math.cos(theta) + 0.5, r * Math.sin(theta) - 0.5 + (Math.random() - 0.5) * 0.5]);
      y.push(1);
    }
    return { X, y };
  }

  function splitData(dataset, trainN) {
    // Shuffle and take trainN for train, rest for test (or use a fixed test set)
    const indices = Array.from({ length: dataset.X.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const trainIdx = indices.slice(0, trainN);
    const testIdx = indices.slice(trainN, Math.min(trainN + 500, indices.length));

    return {
      trainX: trainIdx.map(i => dataset.X[i]),
      trainY: trainIdx.map(i => dataset.y[i]),
      testX: testIdx.map(i => dataset.X[i]),
      testY: testIdx.map(i => dataset.y[i])
    };
  }

  // ── Model (deliberately over-parameterized for small datasets) ───────────
  function buildModel() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 64, inputShape: [2], activation: 'relu' }));
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    return m;
  }

  // ── Chart.js line chart for loss curves ──────────────────────────────────
  let lossChart = null;

  function initLossChart() {
    const ctx = document.getElementById('w4-loss-chart').getContext('2d');
    if (lossChart) lossChart.destroy();
    lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Train Loss (n=20)',   data: [], borderColor: '#e07b6a', backgroundColor: 'rgba(224,123,106,0.08)', borderWidth: 2, pointRadius: 0, tension: 0.4 },
          { label: 'Test Loss  (n=20)',   data: [], borderColor: '#e07b6a', backgroundColor: 'transparent',            borderWidth: 2, borderDash: [6,3], pointRadius: 0, tension: 0.4 },
          { label: 'Train Loss (n=200)',  data: [], borderColor: '#6baed6', backgroundColor: 'rgba(107,174,214,0.08)', borderWidth: 2, pointRadius: 0, tension: 0.4 },
          { label: 'Test Loss  (n=200)',  data: [], borderColor: '#6baed6', backgroundColor: 'transparent',            borderWidth: 2, borderDash: [6,3], pointRadius: 0, tension: 0.4 },
          { label: 'Train Loss (n=2000)', data: [], borderColor: '#52b788', backgroundColor: 'rgba(82,183,136,0.08)',  borderWidth: 2, pointRadius: 0, tension: 0.4 },
          { label: 'Test Loss  (n=2000)', data: [], borderColor: '#52b788', backgroundColor: 'transparent',            borderWidth: 2, borderDash: [6,3], pointRadius: 0, tension: 0.4 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(45,38,64,0.9)',
            titleColor: 'white',
            bodyColor: 'rgba(255,255,255,0.8)',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Epoch', font: { family: 'Inter', size: 11 }, color: '#9b94b0' },
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { font: { family: 'Inter', size: 10 }, color: '#9b94b0', maxTicksLimit: 10 }
          },
          y: {
            title: { display: true, text: 'Loss', font: { family: 'Inter', size: 11 }, color: '#9b94b0' },
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { font: { family: 'Inter', size: 10 }, color: '#9b94b0' },
            min: 0
          }
        }
      }
    });
    return lossChart;
  }

  // ── Gap bar chart ─────────────────────────────────────────────────────────
  let gapChart = null;

  function drawGapChart(gaps) {
    const ctx = document.getElementById('w4-gap-chart').getContext('2d');
    if (gapChart) gapChart.destroy();
    gapChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['n = 20', 'n = 200', 'n = 2000'],
        datasets: [{
          label: 'Generalization Gap (Test Loss − Train Loss)',
          data: gaps,
          backgroundColor: ['rgba(224,123,106,0.7)', 'rgba(107,174,214,0.7)', 'rgba(82,183,136,0.7)'],
          borderColor: ['#e07b6a', '#6baed6', '#52b788'],
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'top', labels: { font: { family: 'Inter', size: 11 }, color: '#6b6480' } },
          tooltip: {
            backgroundColor: 'rgba(45,38,64,0.9)',
            titleColor: 'white',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 10,
            callbacks: { label: ctx => ` Gap: ${ctx.parsed.y.toFixed(4)}` }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Dataset Size', font: { family: 'Inter', size: 11 }, color: '#9b94b0' },
            grid: { display: false },
            ticks: { font: { family: 'Inter', size: 11 }, color: '#6b6480' }
          },
          y: {
            title: { display: true, text: 'Loss Gap', font: { family: 'Inter', size: 11 }, color: '#9b94b0' },
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { font: { family: 'Inter', size: 10 }, color: '#9b94b0' },
            min: 0
          }
        }
      }
    });
  }

  // ── Train at one dataset size ─────────────────────────────────────────────
  async function trainAtSize(trainSize, chart, trainLossDatasetIdx, testLossDatasetIdx, bigDataset) {
    const { trainX, trainY, testX, testY } = splitData(bigDataset, trainSize);

    const xs = tf.tensor2d(trainX);
    const ys = tf.tensor2d(trainY, [trainY.length, 1]);
    const xsTest = tf.tensor2d(testX);
    const ysTest = tf.tensor2d(testY, [testY.length, 1]);

    const model = buildModel();
    const EPOCHS = 150;
    const epochLabels = Array.from({ length: EPOCHS }, (_, i) => i + 1);

    if (chart.data.labels.length === 0) chart.data.labels = epochLabels;

    const trainLosses = [], testLosses = [];

    await model.fit(xs, ys, {
      epochs: EPOCHS,
      batchSize: Math.min(32, trainSize),
      shuffle: true,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          if (stopRequested) {
            model.stopTraining = true;
            return;
          }
          const evalResult = model.evaluate(xsTest, ysTest, { verbose: 0 });
          const testLoss = (await evalResult[0].data())[0];
          evalResult.forEach(t => t.dispose());
          trainLosses.push(logs.loss);
          testLosses.push(testLoss);
        }
      }
    });

    chart.data.datasets[trainLossDatasetIdx].data = trainLosses;
    chart.data.datasets[testLossDatasetIdx].data = testLosses;
    chart.update('none');

    const finalTrain = trainLosses[trainLosses.length - 1];
    const finalTest = testLosses[testLosses.length - 1];

    xs.dispose(); ys.dispose(); xsTest.dispose(); ysTest.dispose();
    return { trainLoss: finalTrain, testLoss: finalTest, gap: finalTest - finalTrain };
  }

  // ── Main training ─────────────────────────────────────────────────────────
  async function train() {
    stopRequested = false;
    const btn = document.getElementById('w4-train-btn');
    btn.disabled = true;
    btn.classList.add('loading');
    document.getElementById('w4-stop-btn').disabled = false;

    const setStatus = (msg) => { document.getElementById('w4-status').textContent = msg; };
    const setProgress = (pct) => {
      const el = document.getElementById('w4-progress');
      if (el) el.style.width = pct + '%';
    };

    const conc = document.getElementById('w4-conclusion');
    if (conc) { conc.style.display = 'none'; conc.innerHTML = ''; }

    setStatus('Generating dataset (2500 samples)…');
    const bigDataset = generateDataset(2500);

    const chart = initLossChart();
    const gaps = [];

    const sizes = [20, 200, 2000];
    const labels = ['n=20', 'n=200', 'n=2000'];
    const accIds = ['w4-acc-20', 'w4-acc-200', 'w4-acc-2000'];
    const gapIds = ['w4-gap-20', 'w4-gap-200', 'w4-gap-2000'];

    for (let s = 0; s < sizes.length; s++) {
      if (stopRequested) break;
      setStatus(`Training at dataset size ${sizes[s]}…`);
      const result = await trainAtSize(sizes[s], chart, s * 2, s * 2 + 1, bigDataset);
      if (stopRequested) break;
      gaps.push(Math.max(0, result.gap));

      // Update accuracy display
      const accEl = document.getElementById(accIds[s]);
      if (accEl) {
        accEl.innerHTML = `
          <div class="label">${labels[s]}</div>
          <div class="value" style="font-size:18px;">${result.trainLoss.toFixed(3)}</div>
          <div class="sublabel">train loss</div>
          <div style="font-size:14px;font-weight:700;color:#e07b6a;margin-top:4px;">${result.testLoss.toFixed(3)}</div>
          <div class="sublabel">test loss</div>
        `;
      }
      const gapEl = document.getElementById(gapIds[s]);
      if (gapEl) {
        gapEl.innerHTML = `
          <div class="label">Gap</div>
          <div class="value" style="font-size:18px;${result.gap < 0.05 ? 'color:#52b788' : 'color:#e07b6a'}">${Math.max(0, result.gap).toFixed(3)}</div>
        `;
      }

      // Draw network viz after final training size (n=2000) with live weights
      if (s === sizes.length - 1) {
        // Re-train a fresh model just for viz purposes (reuse last bigDataset split)
        const vizSplit = splitData(bigDataset, sizes[s]);
        const xsViz = tf.tensor2d(vizSplit.trainX);
        const ysViz = tf.tensor2d(vizSplit.trainY, [vizSplit.trainY.length, 1]);
        const vizModel = buildModel();
        await vizModel.fit(xsViz, ysViz, { epochs: 50, batchSize: 64, verbose: 0 });
        const wMats = await NetworkViz.extractWeights(vizModel);
        NetworkViz.draw(document.getElementById('w4-network'), LAYERS_MODEL, wMats);
        xsViz.dispose(); ysViz.dispose();
      }

      setProgress((s + 1) / sizes.length * 100);
    }

    if (!stopRequested) {
      drawGapChart(gaps);

      setStatus('✓ Training complete — the gap shrinks as data grows!');
      setProgress(100);
      btn.disabled = false;
      btn.classList.remove('loading');
      document.getElementById('w4-stop-btn').disabled = true;

      if (conc && gaps.length >= 3) {
        conc.style.display = 'block';
        conc.innerHTML = `<strong>Conclusion:</strong> With n=20, the model easily memorized the data but failed to generalize, leaving a large generalization gap of <strong>${gaps[0].toFixed(3)}</strong>. As the dataset grew to n=2000, the gap shrank to <strong>${gaps[2].toFixed(3)}</strong>, proving that larger datasets force the model to learn generalizable patterns.`;
      }
    }
  }

  return { init: () => {
    // Draw skeleton network diagram immediately
    NetworkViz.draw(document.getElementById('w4-network'), LAYERS_MODEL, null);

    document.getElementById('w4-train-btn').addEventListener('click', train);
    
    document.getElementById('w4-stop-btn').addEventListener('click', () => {
      stopRequested = true;
      document.getElementById('w4-stop-btn').disabled = true;
      document.getElementById('w4-status').textContent = 'Training stopped.';
      document.getElementById('w4-train-btn').disabled = false;
      document.getElementById('w4-train-btn').classList.remove('loading');
    });

    document.getElementById('w4-reset-btn').addEventListener('click', () => {
      stopRequested = true;
      document.getElementById('w4-stop-btn').disabled = true;
      document.getElementById('w4-train-btn').disabled = false;
      document.getElementById('w4-train-btn').classList.remove('loading');
      document.getElementById('w4-status').textContent = 'Click Train to begin (trains n=20, 200, 2000)';
      document.getElementById('w4-progress').style.width = '0%';
      
      const conc = document.getElementById('w4-conclusion');
      if (conc) { conc.style.display = 'none'; conc.innerHTML = ''; }
      
      const accIds = ['w4-acc-20', 'w4-acc-200', 'w4-acc-2000'];
      const gapIds = ['w4-gap-20', 'w4-gap-200', 'w4-gap-2000'];
      const labels = ['n = 20', 'n = 200', 'n = 2000'];
      const colors = ['#e07b6a', '#6baed6', '#52b788'];
      
      for (let s = 0; s < accIds.length; s++) {
        const accEl = document.getElementById(accIds[s]);
        if (accEl) {
          accEl.innerHTML = `
            <div class="label">${labels[s]}</div>
            <div class="value" style="font-size:18px;background:linear-gradient(135deg,${colors[s]},${colors[s]});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">—</div>
            <div class="sublabel">train / test loss</div>
          `;
        }
        const gapEl = document.getElementById(gapIds[s]);
        if (gapEl) {
          gapEl.innerHTML = `
            <div class="label">Gap (${labels[s].replace(/\s/g, '')})</div>
            <div class="value" style="font-size:18px;color:${colors[s]};">—</div>
          `;
        }
      }
      
      initLossChart();
      const ctx = document.getElementById('w4-gap-chart').getContext('2d');
      ctx.clearRect(0, 0, 400, 400); // Clear the gap chart canvas
      if (gapChart) {
        gapChart.destroy();
        gapChart = null;
      }
      NetworkViz.draw(document.getElementById('w4-network'), LAYERS_MODEL, null);
    });
  }};
})();
