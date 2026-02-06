// ui/app.js

const CSV_PATH = "../forecasts/forecast_daily_summary.csv"; // relative to /ui/

let chart = null;
let allRows = [];

const $ = (id) => document.getElementById(id);

function toNum(x) {
  const v = Number(String(x ?? "").trim());
  return Number.isFinite(v) ? v : null;
}

function formatNum(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "";
  return x.toFixed(2);
}

function parseCSV(text) {
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  if (parsed.errors?.length) console.warn("CSV parse errors:", parsed.errors);

  const rows = (parsed.data || [])
    .map((r) => ({
      Date: (r.Date || "").trim(),
      PredictionToday: toNum(r.PredictionToday),
      PredictionTomorrow: toNum(r.PredictionTomorrow),
    }))
    .filter((r) => r.Date);

  // Keep in ascending order for time-series
  rows.sort((a, b) => a.Date.localeCompare(b.Date));
  return rows;
}

function updateMeta(rows) {
  const meta = $("meta");
  if (!meta) return;

  if (!rows.length) {
    meta.textContent = " (žádná data)";
    return;
  }
  const first = rows[0].Date;
  const last = rows[rows.length - 1].Date;
  meta.textContent = ` • ${rows.length} řádků • ${first} → ${last}`;
}

function getFilteredRowsForTable() {
  const q = $("searchInput")?.value?.trim() ?? "";
  const desc = $("descToggle")?.checked ?? true;

  let rows = allRows;
  if (q) rows = rows.filter((r) => r.Date.includes(q));

  rows = rows.slice();
  rows.sort((a, b) => (desc ? b.Date.localeCompare(a.Date) : a.Date.localeCompare(b.Date)));
  return rows;
}

function sliceByDays(rowsAsc) {
  const n = Number($("daysSelect")?.value ?? 30);
  if (!n || n <= 0) return rowsAsc;
  return rowsAsc.slice(Math.max(0, rowsAsc.length - n));
}

function renderTable() {
  const tbody = $("tbody");
  if (!tbody) return;

  const rows = getFilteredRowsForTable();
  tbody.innerHTML = "";

  for (const r of rows) {
    const delta =
      (r.PredictionTomorrow ?? 0) - (r.PredictionToday ?? 0);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.Date}</td>
      <td class="num">${formatNum(r.PredictionToday ?? NaN)}</td>
      <td class="num">${formatNum(r.PredictionTomorrow ?? NaN)}</td>
      <td class="num">${formatNum(delta)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderChart() {
  const canvas = $("chart");
  if (!canvas) return;

  // Always chart in ASC time order
  const rowsAsc = allRows.slice().sort((a, b) => a.Date.localeCompare(b.Date));
  const rows = sliceByDays(rowsAsc);

  const labels = rows.map((r) => r.Date);
  const today = rows.map((r) => r.PredictionToday ?? 0);
  const tomorrow = rows.map((r) => r.PredictionTomorrow ?? 0);

  if (chart) {
    chart.destroy();
    chart = null;
  }

  chart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "PredictionToday (kWh)", data: today, tension: 0.25 },
        { label: "PredictionTomorrow (kWh)", data: tomorrow, tension: 0.25 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,

      // Helps avoid “growing canvas” layout feedback loops
      animation: false,
      resizeDelay: 100,

      plugins: {
        legend: { labels: { color: "#9fb0c3" } },
        tooltip: { enabled: true },
      },
      scales: {
        x: {
          ticks: { color: "#9fb0c3", maxRotation: 0 },
          grid: { color: "rgba(34,48,67,.4)" },
        },
        y: {
          ticks: { color: "#9fb0c3" },
          grid: { color: "rgba(34,48,67,.4)" },
        },
      },
    },
  });
}

async function loadData() {
  // cache-bust so you see fresh CSV after actions run
  const url = `${CSV_PATH}?v=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch CSV: ${res.status} ${res.statusText}`);
  const text = await res.text();

  allRows = parseCSV(text);
  updateMeta(allRows);

  renderChart();
  renderTable();
}

function wireUI() {
  $("reloadBtn")?.addEventListener("click", () => loadData().catch((e) => alert(e.message)));
  $("searchInput")?.addEventListener("input", renderTable);
  $("descToggle")?.addEventListener("change", renderTable);
  $("daysSelect")?.addEventListener("change", renderChart);

  // Re-render chart on viewport resize (throttled by Chart.js resizeDelay)
  window.addEventListener("resize", () => {
    if (!chart) return;
    // Chart.js handles this, but explicit update can help on some browsers
    try {
      chart.resize();
      chart.update("none");
    } catch (_) {}
  });
}

wireUI();
loadData().catch((e) => {
  console.error(e);
  alert(
    "Nepodařilo se načíst CSV. Zkontroluj, že existuje forecasts/forecast_daily_summary.csv a že je commitnutý."
  );
});
