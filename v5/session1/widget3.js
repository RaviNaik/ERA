/* ===== WIDGET 3: Embeddings Learn Similarity from Next-Token ===== */

const W3 = (() => {
  // ── Vocabulary & Grammar ─────────────────────────────────────────────────
  const VOCAB = {
    // categories
    animals: ['cat', 'dog', 'cow'],
    fruits:  ['apple', 'mango'],
    verbs:   ['eat', 'chase', 'see'],
    // structural tokens
    special: ['<S>', '<E>']
  };

  const ALL_TOKENS = [
    ...VOCAB.animals, ...VOCAB.fruits, ...VOCAB.verbs, ...VOCAB.special
  ];
  // '<S>'=10, '<E>'=11
  const TOKEN2ID = {};
  ALL_TOKENS.forEach((t, i) => { TOKEN2ID[t] = i; });
  const VOCAB_SIZE = ALL_TOKENS.length;  // 10 tokens

  const CATEGORY_COLORS = {
    cat: '#e07b6a', dog: '#e07b6a', cow: '#e07b6a',
    apple: '#6baed6', mango: '#6baed6',
    eat: '#52b788', chase: '#52b788', see: '#52b788',
    '<S>': '#c8a8e0', '<E>': '#c8a8e0'
  };
  const CATEGORY_LABELS = {
    cat: 'Animal', dog: 'Animal', cow: 'Animal',
    apple: 'Fruit', mango: 'Fruit',
    eat: 'Verb', chase: 'Verb', see: 'Verb',
    '<S>': 'Special', '<E>': 'Special'
  };

  // ── Sentence templates ────────────────────────────────────────────────────
  // Template: <S> [animal] [verb] [fruit] <E>
  // This means each animal shares next-token distribution with other animals
  function generateCorpus(n = 2000) {
    const sentences = [];
    for (let i = 0; i < n; i++) {
      const animal = VOCAB.animals[Math.floor(Math.random() * VOCAB.animals.length)];
      const verb   = VOCAB.verbs[Math.floor(Math.random() * VOCAB.verbs.length)];
      const fruit  = VOCAB.fruits[Math.floor(Math.random() * VOCAB.fruits.length)];
      sentences.push(['<S>', animal, verb, fruit, '<E>']);
    }
    return sentences;
  }

  function corpusToBigrams(sentences) {
    // Return array of [input_token_id, target_token_id]
    const pairs = [];
    for (const sent of sentences) {
      for (let i = 0; i < sent.length - 1; i++) {
        pairs.push([TOKEN2ID[sent[i]], TOKEN2ID[sent[i + 1]]]);
      }
    }
    return pairs;
  }

  // ── Embedding model (embedding → softmax next-token) ──────────────────────
  function buildEmbeddingModel(embedDim = 8) {
    const input = tf.input({ shape: [1] });
    // Embedding lookup: vocab_size → embedDim
    const emb = tf.layers.embedding({ inputDim: VOCAB_SIZE, outputDim: embedDim, name: 'embedding' })(input);
    const flat = tf.layers.flatten()(emb);
    const output = tf.layers.dense({ units: VOCAB_SIZE, activation: 'softmax' })(flat);
    const model = tf.model({ inputs: input, outputs: output });
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  // ── PCA: project nxD embeddings to 2D ────────────────────────────────────
  function pca2D(matrix) {
    // matrix: [n, d] float array
    const n = matrix.length, d = matrix[0].length;
    // Center
    const mean = Array(d).fill(0);
    matrix.forEach(row => row.forEach((v, j) => { mean[j] += v / n; }));
    const centered = matrix.map(row => row.map((v, j) => v - mean[j]));

    // Power iteration for 2 PCs
    function powerIter(data, k = 50) {
      let v = Array.from({ length: d }, () => Math.random() - 0.5);
      for (let i = 0; i < k; i++) {
        // v = Cov @ v (approximated as X^T X v / n)
        const Xv = data.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
        const newV = Array(d).fill(0);
        data.forEach((row, i2) => row.forEach((x, j) => { newV[j] += x * Xv[i2] / n; }));
        const norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
        v = newV.map(x => x / norm);
      }
      return v;
    }

    const pc1 = powerIter(centered);
    // Deflate
    const scores1 = centered.map(row => row.reduce((s, x, j) => s + x * pc1[j], 0));
    const deflated = centered.map((row, i) => row.map((x, j) => x - scores1[i] * pc1[j]));
    const pc2 = powerIter(deflated);
    const scores2 = deflated.map(row => row.reduce((s, x, j) => s + x * pc2[j], 0));

    return scores1.map((s1, i) => [s1, scores2[i]]);
  }

  // ── Draw embedding scatter plot on canvas ────────────────────────────────
  function drawEmbeddingPlot(canvas, coords) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const xs = coords.map(c => c[0]), ys = coords.map(c => c[1]);
    const minX = Math.min(...xs) - 0.5, maxX = Math.max(...xs) + 0.5;
    const minY = Math.min(...ys) - 0.5, maxY = Math.max(...ys) + 0.5;

    const toPixX = x => 40 + ((x - minX) / (maxX - minX)) * (W - 80);
    const toPixY = y => H - 40 - ((y - minY) / (maxY - minY)) * (H - 80);

    // Background grid
    ctx.strokeStyle = 'rgba(100,90,120,0.1)'; ctx.lineWidth = 0.5;
    for (let g = 0; g <= 5; g++) {
      const gx = 40 + g * (W - 80) / 5;
      const gy = 40 + g * (H - 80) / 5;
      ctx.beginPath(); ctx.moveTo(gx, 40); ctx.lineTo(gx, H - 40); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(40, gy); ctx.lineTo(W - 40, gy); ctx.stroke();
    }

    // Draw connecting lines within categories
    const categories = {
      animals: VOCAB.animals.map(t => ALL_TOKENS.indexOf(t)),
      fruits:  VOCAB.fruits.map(t => ALL_TOKENS.indexOf(t)),
      verbs:   VOCAB.verbs.map(t => ALL_TOKENS.indexOf(t)),
    };
    const catColors = { animals: '#e07b6a', fruits: '#6baed6', verbs: '#52b788' };

    Object.entries(categories).forEach(([cat, ids]) => {
      if (ids.length < 2) return;
      ctx.strokeStyle = catColors[cat] + '55';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          ctx.beginPath();
          ctx.moveTo(toPixX(coords[ids[i]][0]), toPixY(coords[ids[i]][1]));
          ctx.lineTo(toPixX(coords[ids[j]][0]), toPixY(coords[ids[j]][1]));
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    });

    // Draw tokens
    ALL_TOKENS.forEach((token, idx) => {
      const [cx, cy] = coords[idx];
      const px = toPixX(cx), py = toPixY(cy);
      const color = CATEGORY_COLORS[token];

      // Shadow
      ctx.shadowColor = color + '55';
      ctx.shadowBlur = 8;

      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = '#2d2640';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(token, px, py + 13);
    });

    // Axes labels
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#9b94b0';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PC 1', W / 2, H - 12);
    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('PC 2', 0, 0);
    ctx.restore();
  }

  // ── Nearest-neighbor table ────────────────────────────────────────────────
  function buildNNTable(embeddingMatrix) {
    const tokens = ALL_TOKENS.slice(0, 8); // exclude special
    const rows = [];

    tokens.forEach((t, i) => {
      const vec = embeddingMatrix[i];
      const dists = ALL_TOKENS.slice(0, 8).map((t2, j) => {
        const v2 = embeddingMatrix[j];
        const dot = vec.reduce((s, x, k) => s + x * v2[k], 0);
        const n1 = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
        const n2 = Math.sqrt(v2.reduce((s, x) => s + x * x, 0));
        return { token: t2, sim: dot / (n1 * n2 + 1e-9) };
      });
      dists.sort((a, b) => b.sim - a.sim);
      const nn = dists.filter(d => d.token !== t).slice(0, 2);
      rows.push({ token: t, nn, cat: CATEGORY_LABELS[t] });
    });

    const el = document.getElementById('w3-nn-table');
    if (!el) return;
    el.innerHTML = `<tr>
      <th>Token</th><th>Category</th><th>Nearest Neighbor</th><th>2nd Neighbor</th>
    </tr>` + rows.map(r => `<tr>
      <td><span class="token-badge">
        <span class="legend-dot" style="background:${CATEGORY_COLORS[r.token]}"></span>
        <strong>${r.token}</strong>
      </span></td>
      <td>${r.cat}</td>
      <td><span class="token-badge">
        <span class="legend-dot" style="background:${CATEGORY_COLORS[r.nn[0].token]}"></span>
        ${r.nn[0].token}
      </span> <small style="color:#9b94b0">(${r.nn[0].sim.toFixed(3)})</small></td>
      <td><span class="token-badge">
        <span class="legend-dot" style="background:${CATEGORY_COLORS[r.nn[1].token]}"></span>
        ${r.nn[1].token}
      </span> <small style="color:#9b94b0">(${r.nn[1].sim.toFixed(3)})</small></td>
    </tr>`).join('');
  }

  // ── Main training ─────────────────────────────────────────────────────────
  async function train() {
    const btn = document.getElementById('w3-train-btn');
    btn.disabled = true;
    btn.classList.add('loading');

    const setStatus = (msg) => { document.getElementById('w3-status').textContent = msg; };
    const setProgress = (pct) => {
      const el = document.getElementById('w3-progress');
      if (el) el.style.width = pct + '%';
    };

    setStatus('Generating corpus…');
    const corpus = generateCorpus(2000);
    const bigrams = corpusToBigrams(corpus);

    const inputIds = bigrams.map(b => b[0]);
    const targetIds = bigrams.map(b => b[1]);
    const xs = tf.tensor2d(inputIds, [inputIds.length, 1], 'int32');
    const ys = tf.tensor1d(targetIds, 'int32');

    setStatus('Training embedding model…');
    const model = buildEmbeddingModel(8);

    const EPOCHS = 30;
    await model.fit(xs, ys, {
      epochs: EPOCHS,
      batchSize: 256,
      shuffle: true,
      callbacks: {
        onEpochEnd: (ep, logs) => {
          setProgress((ep + 1) / EPOCHS * 100);
          setStatus(`Epoch ${ep + 1}/${EPOCHS} — loss: ${logs.loss.toFixed(4)}, acc: ${(logs.acc * 100).toFixed(1)}%`);
        }
      }
    });

    xs.dispose(); ys.dispose();

    // Extract embedding table
    const embLayer = model.getLayer('embedding');
    const embWeights = embLayer.getWeights()[0];
    const embData = await embWeights.data();
    const embedDim = embWeights.shape[1];

    const embMatrix = [];
    for (let i = 0; i < VOCAB_SIZE; i++) {
      embMatrix.push(Array.from(embData.slice(i * embedDim, (i + 1) * embedDim)));
    }

    // PCA to 2D
    const coords2D = pca2D(embMatrix);

    // Draw
    const canvas = document.getElementById('w3-canvas');
    drawEmbeddingPlot(canvas, coords2D);
    buildNNTable(embMatrix);

    setStatus('✓ Embeddings learned! Same-category tokens cluster together.');
    setProgress(100);
    btn.disabled = false;
    btn.classList.remove('loading');
  }

  return { init: () => {
    document.getElementById('w3-train-btn').addEventListener('click', train);
  }};
})();
