/* ===== Network Architecture Visualizer =====
 * Draws a neural net diagram with:
 *   - Nodes: input (blue), hidden (gold), output (green)
 *   - Connections: coral = positive weight, blue = negative weight
 *   - Line width ∝ |weight|
 * Usage:
 *   NetworkViz.drawEmpty(canvas, layerDefs)          — skeleton before training
 *   NetworkViz.draw(canvas, layerDefs, weightMats)   — live weights after training
 *   const mats = await NetworkViz.extractWeights(model)
 */
const NetworkViz = (() => {
  const MAX_NODES = 8;       // max nodes shown per layer
  const NODE_R    = 13;      // node radius px
  const PAD_X     = 54;      // horizontal padding
  const PAD_TOP   = 24;      // top padding
  const PAD_BOT   = 44;      // bottom padding (space for layer labels + legend)

  // ── Colour palette ─────────────────────────────────────────────────────────
  const COL = {
    input:   '#6baed6',
    hidden:  '#f0a04a',
    output:  '#52b788',
    pos:     (a) => `rgba(224,107,90,${a})`,   // coral — positive
    neg:     (a) => `rgba(90,160,214,${a})`,   // blue  — negative
    zero:    (a) => `rgba(170,160,190,${a})`,  // grey  — ~zero / unknown
    bg:      'rgba(255,255,255,0.55)',
    axis:    'rgba(100,90,120,0.18)',
    label:   '#9b94b0',
    sublabel:'#6b6480',
  };

  // ── Layout helpers ──────────────────────────────────────────────────────────
  function nodePositions(canvas, layerDefs) {
    const W = canvas.width, H = canvas.height;
    const usableW = W - PAD_X * 2;
    const usableH = H - PAD_TOP - PAD_BOT;
    const nL = layerDefs.length;

    return layerDefs.map((ld, li) => {
      const x = nL === 1 ? W / 2 : PAD_X + (li / (nL - 1)) * usableW;
      const nVis = Math.min(ld.nodes, MAX_NODES);
      const spacing = usableH / (nVis + 1);
      const ys = Array.from({ length: nVis }, (_, ni) => PAD_TOP + (ni + 1) * spacing);
      return { x, ys, ld };
    });
  }

  // ── Draw connections ────────────────────────────────────────────────────────
  function drawConnections(ctx, layers, weights) {
    for (let li = 0; li < layers.length - 1; li++) {
      const from = layers[li], to = layers[li + 1];
      const wMat = weights ? weights[li] : null; // [inDim, outDim]

      // normalise weights for visual scale
      let maxAbs = 0;
      if (wMat) {
        for (const row of wMat) for (const v of row) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
      if (maxAbs === 0) maxAbs = 1;

      for (let fi = 0; fi < from.ys.length; fi++) {
        for (let ti = 0; ti < to.ys.length; ti++) {
          let w = 0;
          if (wMat) {
            // Map visible node index back to real weight index
            const origFi = Math.round(fi * (from.ld.nodes - 1) / Math.max(from.ys.length - 1, 1));
            const origTi = Math.round(ti * (to.ld.nodes - 1) / Math.max(to.ys.length - 1, 1));
            w = (wMat[origFi] !== undefined && wMat[origFi][origTi] !== undefined)
              ? wMat[origFi][origTi] : 0;
          }

          const absNorm = wMat ? Math.abs(w) / maxAbs : 0;
          const alpha = wMat ? (0.08 + absNorm * 0.72) : 0.07;
          const lw    = wMat ? (0.4  + absNorm * 3.2)  : 0.8;

          ctx.beginPath();
          ctx.moveTo(from.x, from.ys[fi]);
          ctx.lineTo(to.x,   to.ys[ti]);
          ctx.lineWidth = lw;
          if (!wMat) {
            ctx.strokeStyle = COL.zero(alpha);
          } else if (w >= 0) {
            ctx.strokeStyle = COL.pos(alpha);
          } else {
            ctx.strokeStyle = COL.neg(alpha);
          }
          ctx.stroke();
        }
      }
    }
  }

  // ── Draw nodes ──────────────────────────────────────────────────────────────
  function drawNodes(ctx, layers) {
    layers.forEach(({ x, ys, ld }) => {
      const color = ld.type === 'input'  ? COL.input  :
                    ld.type === 'output' ? COL.output  : COL.hidden;

      ys.forEach((y, ni) => {
        // Glow
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur  = 10;
        ctx.beginPath();
        ctx.arc(x, y, NODE_R, 0, Math.PI * 2);
        ctx.fillStyle   = 'white';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2.5;
        ctx.stroke();
        ctx.restore();

        // Node label (e.g. x₁, x₂ for input; ŷ for output)
        const lbl = ld.nodeLabels ? (ld.nodeLabels[ni] || '') : '';
        if (lbl) {
          ctx.font         = `bold 10px Inter, sans-serif`;
          ctx.fillStyle    = color;
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(lbl, x, y);
        }

        // Activation pill inside node (small text, no label)
        if (ld.activation && !lbl) {
          ctx.font         = '8px Inter, sans-serif';
          ctx.fillStyle    = color + 'cc';
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(ld.activation === 'relu' ? 'R' : ld.activation === 'sigmoid' ? 'σ' : '', x, y);
        }
      });

      // Truncation badge: if actual nodes > MAX_NODES
      if (ld.nodes > MAX_NODES) {
        const midY = (ys[Math.floor(ys.length / 2) - 1] + ys[Math.floor(ys.length / 2)]) / 2;
        ctx.font         = '10px Inter, sans-serif';
        ctx.fillStyle    = COL.label;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`(${ld.nodes})`, x, midY);
      }
    });
  }

  // ── Draw layer labels ───────────────────────────────────────────────────────
  function drawLayerLabels(ctx, canvas, layers) {
    const H = canvas.height;
    layers.forEach(({ x, ld }) => {
      ctx.font         = '11px Inter, sans-serif';
      ctx.fillStyle    = COL.sublabel;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(ld.label || '', x, H - PAD_BOT + 8);
      if (ld.activation) {
        ctx.font      = '10px Inter, sans-serif';
        ctx.fillStyle = COL.label;
        const actName = { relu: 'ReLU', sigmoid: 'Sigmoid', linear: 'Linear', softmax: 'Softmax' }[ld.activation] || ld.activation;
        ctx.fillText(actName, x, H - PAD_BOT + 22);
      }
    });
  }

  // ── Draw legend ─────────────────────────────────────────────────────────────
  function drawLegend(ctx, canvas) {
    const H = canvas.height, W = canvas.width;
    const y = H - 14;
    ctx.font         = '10px Inter, sans-serif';
    ctx.textBaseline = 'middle';

    const items = [
      { color: COL.pos(0.9), label: 'positive weight' },
      { color: COL.neg(0.9), label: 'negative weight' },
    ];

    let cx = 10;
    items.forEach(item => {
      ctx.beginPath();
      ctx.moveTo(cx, y); ctx.lineTo(cx + 20, y);
      ctx.strokeStyle = item.color; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.fillStyle   = COL.label;
      ctx.textAlign   = 'left';
      ctx.fillText(item.label, cx + 24, y);
      cx += 24 + ctx.measureText(item.label).width + 18;
    });
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Draw the network (optionally with live weights).
   * @param {HTMLCanvasElement} canvas
   * @param {Array}  layerDefs  [{nodes, label, type, activation, nodeLabels?}, ...]
   * @param {Array|null} weightMats  array of 2-D number arrays [inDim][outDim], one per connection
   */
  function draw(canvas, layerDefs, weightMats = null) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layers = nodePositions(canvas, layerDefs);
    drawConnections(ctx, layers, weightMats);
    drawNodes(ctx, layers);
    drawLayerLabels(ctx, canvas, layers);
    drawLegend(ctx, canvas);
  }

  /**
   * Extract weight matrices from a TF.js Sequential model.
   * Returns array of 2-D JS arrays, one per dense layer (kernel only).
   */
  async function extractWeights(model) {
    const mats = [];
    for (const layer of model.layers) {
      const ws = layer.getWeights();
      if (ws.length === 0) continue;
      const kernel = ws[0];                    // shape [inDim, outDim]
      const data   = await kernel.data();
      const [inD, outD] = kernel.shape;
      const mat = [];
      for (let i = 0; i < inD; i++) {
        const row = [];
        for (let j = 0; j < outD; j++) row.push(data[i * outD + j]);
        mat.push(row);
      }
      mats.push(mat);
    }
    return mats;
  }

  return { draw, extractWeights };
})();
