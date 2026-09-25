(function () {
  const data = window.SESSION_DATA || { heroStats: [], runs: [], findings: [] };

  document.getElementById("heroStats").innerHTML = data.heroStats.map((item) => `
    <article><strong>${item.value}</strong><span>${item.label}</span></article>
  `).join("");

  const table = document.getElementById("runsTable");
  if (!data.runs.length) {
    table.innerHTML = "<tbody><tr><td>No runs exported yet.</td></tr></tbody>";
  } else {
    table.innerHTML = `
      <thead><tr><th>Run</th><th>Variant</th><th>Batch</th><th>Final Val Loss</th><th>Tokens/s</th><th>Peak MB</th></tr></thead>
      <tbody>${data.runs.map((run) => `
        <tr><td>${run.label}</td><td>${run.variant}</td><td>${run.batch_size}</td><td>${run.final_val_loss.toFixed(4)}</td><td>${Math.round(run.mean_tokens_per_sec).toLocaleString()}</td><td>${Math.round(run.peak_memory_mb).toLocaleString()}</td></tr>
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
  const maxMem = Math.max(...data.runs.map((run) => run.peak_memory_mb), 1);
  const barWidth = canvas.clientWidth / (data.runs.length * 2 + 1);
  data.runs.forEach((run, index) => {
    const x = barWidth * (index * 2 + 1);
    const h = (run.peak_memory_mb / maxMem) * 210;
    ctx.fillStyle = "#2f6fed";
    ctx.fillRect(x, 240 - h, barWidth, h);
    ctx.fillStyle = "#111827";
    ctx.font = "12px sans-serif";
    ctx.fillText(run.variant, x, 260);
  });
})();