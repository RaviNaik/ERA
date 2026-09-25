(function () {
  const data = window.SESSION_DATA || { heroStats: [], runs: [], findings: [] };

  document.getElementById("heroStats").innerHTML = data.heroStats.map((item) => `
    <article><strong>${item.value}</strong><span>${item.label}</span></article>
  `).join("");

  const table = document.getElementById("runsTable");
  if (!data.runs.length) {
    table.innerHTML = "<tbody><tr><td>No runs exported yet.</td></tr></tbody>";
  } else {
    const num = (v, digits = 0) => (v === null || v === undefined) ? "n/a"
      : Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    const pct = (v) => (v === null || v === undefined) ? "n/a" : `${(v * 100).toFixed(1)}%`;
    table.innerHTML = `
      <thead><tr><th>Run</th><th>Variant</th><th>Status</th><th>Batch</th><th>Steps</th><th>Val Loss</th><th>Val Acc</th><th>Tokens/s</th><th>MFU</th><th>Peak MB</th><th>Act MB/seq</th></tr></thead>
      <tbody>${data.runs.map((run) => `
        <tr><td>${run.label}</td><td>${run.variant}</td><td>${run.status}</td><td>${run.batch_size}</td><td>${num(run.total_steps)}</td><td>${num(run.final_val_loss, 4)}</td><td>${pct(run.final_val_accuracy)}</td><td>${num(run.mean_tokens_per_sec)}</td><td>${pct(run.mfu)}</td><td>${num(run.peak_step_memory_mb)}</td><td>${num(run.activation_saved_mb_per_sample, 2)}</td></tr>
      `).join("")}</tbody>`;
  }

  document.getElementById("findings").innerHTML = data.findings.map((item) => `<li>${item}</li>`).join("");

  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * devicePixelRatio;
  const height = canvas.height = 280 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, width, height);
  if (!data.runs.length) return;
  const runs = data.runs.filter((run) => run.peak_step_memory_mb);
  const maxMem = Math.max(...runs.map((run) => run.peak_step_memory_mb), 1);
  const barWidth = canvas.clientWidth / (runs.length * 2 + 1);
  runs.forEach((run, index) => {
    const x = barWidth * (index * 2 + 1);
    const h = (run.peak_step_memory_mb / maxMem) * 210;
    ctx.fillStyle = "#2f6fed";
    ctx.fillRect(x, 240 - h, barWidth, h);
    ctx.fillStyle = "#111827";
    ctx.font = "12px sans-serif";
    ctx.fillText(run.label, x, 260);
  });
})();