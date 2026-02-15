/* ui/app.js
 * Dashboard for forecasts/forecast_daily_summary.csv
 * Columns: Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
 *
 * index.html IDs:
 * - canvas#chart
 * - tbody#tbody
 * - select#daysSelect
 * - button#reloadBtn
 * - input#searchInput
 * - input#descToggle
 * - span#meta
 */

let chartInstance = null;
let allRows = [];

const $ = (id) => document.getElementById(id);

function parseNumber(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number.parseFloat(s.replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

function stripBOM(s) {
  if (!s) return s;
  return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s;
}

async function fetchCSV() {
  // /FVE/ui/ -> ../forecasts/... -> /FVE/forecasts/...
  const url = new URL("../forecasts/forecast_daily_summary.csv", window.location.href);
  url.searchParams.set("_", String(Date.now())); // cache-bust
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to fetch CSV: HTTP ${r.status}`);
  return await r.text();
}

function parseCSV(text) {
  const lines = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter((l) => l.trim().length > 0);

  if (lines.length < 2) return [];

  const header = lines[0].split(",").map((h) => stripBOM(h.trim()));
  const idx = (name) => header.indexOf(name);

  const iDate = idx("Date");
  const iPredToday = idx("PredictionToday");
  const iActual = idx("ActualToday");
  const iPredTomorrow = idx("PredictionTomorrow");

  if (iDate === -1 || iPredToday === -1 || iPredTomorrow === -1) {
    console.warn("CSV header:", header);
    throw new Error("CSV missing required columns: Date, PredictionToday, PredictionTomorrow");
  }

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const d = (cols[iDate] ?? "").trim();
    if (!d) continue;

    rows.push({
      Date: d,
      PredictionToday: parseNumber(cols[iPredToday]),
      ActualToday: iActual !== -1 ? parseNumber(cols[iActual]) : null,
      PredictionTomorrow: parseNumber(cols[iPredTomorrow]),
    });
  }

  rows.sort((a, b) => a.Date.localeCompare(b.Date));
  return rows;
}

function applyFilters(rows) {
  const search = ($("searchInput")?.value || "").trim();
  const desc = $("descToggle")?.checked ?? true;

  let out = rows;

  if (search) out = out.filter((r) => r.Date.includes(search));

  // table order
  out = out.slice().sort((a, b) => (desc ? b.Date.localeCompare(a.Date) : a.Date.localeCompare(b.Date)));

  // days
  const nDays = $("daysSelect") ? Number.parseInt($("daysSelect").value, 10) : 30;
  if (Number.isFinite(nDays) && nDays > 0 && out.length > nDays) out = out.slice(0, nDays);

  // chart needs chronological
  const chartRows = out.slice().sort((a, b) => a.Date.localeCompare(b.Date));

  return { tableRows: out, chartRows };
}

function computeSuggestedMax(values) {
  const nums = values.filter((v) => Number.isFinite(v));
  if (nums.length === 0) return 5;
  const m = Math.max(...nums);
  const head = m * 1.15 + 0.5;
  return Math.ceil(head * 2) / 2; // 0.5 steps
}

function ensureFixedChartHeight() {
  const canvas = $("chart");
  if (!canvas) return;
  const wrap = canvas.parentElement; // .chart-wrap
  if (wrap) {
    wrap.style.height = "420px";
    wrap.style.minHeight = "420px";
    wrap.style.maxHeight = "420px";
  }
  canvas.style.height = "420px";
}

function renderChart(rows) {
  const canvas = $("chart");
  if (!canvas) return;

  ensureFixedChartHeight();

  if (typeof Chart === "undefined") {
    throw new Error("Chart.js is not loaded (Chart is undefined).");
  }

  // IMPORTANT: destroy any existing chart bound to this canvas (even from previous app.js versions)
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
  if (chartInstance) {
    try { chartInstance.destroy(); } catch (_) {}
    chartInstance = null;
  }

  const labels = rows.map((r) => r.Date);
  const predToday = rows.map((r) => r.PredictionToday);
  const predTomorrow = rows.map((r) => r.PredictionTomorrow);
  const actualToday = rows.map((r) => r.ActualToday);

  const suggestedMax = computeSuggestedMax([...predToday, ...predTomorrow, ...actualToday]);

  chartInstance = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "PredictionToday (kWh)",
          data: predToday,
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.25,
          spanGaps: true,
        },
        {
          label: "PredictionTomorrow (kWh)",
          data: predTomorrow,
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.25,
          spanGaps: true,
        },
        {
          label: "ActualToday (kWh)",
          data: actualToday,
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.25,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: { beginAtZero: true, suggestedMax },
        x: { ticks: { autoSkip: true, maxTicksLimit: 12, maxRotation: 0 } },
      },
      plugins: {
        legend: { display: true },
        tooltip: { intersect: false, mode: "index" },
      },
      interaction: { intersect: false, mode: "index" },
    },
  });
}

function renderTable(rows) {
  const tbody = $("tbody");
  if (!tbody) return;

  tbody.innerHTML = "";

  for (const r of rows) {
    const tr = document.createElement("tr");

    const tdDate = document.createElement("td");
    tdDate.textContent = r.Date;
    tr.appendChild(tdDate);

    const tdP1 = document.createElement("td");
    tdP1.className = "num";
    tdP1.textContent = r.PredictionToday == null ? "" : r.PredictionToday.toFixed(2);
    tr.appendChild(tdP1);

    const tdA = document.createElement("td");
    tdA.className = "num";
    tdA.textContent = r.ActualToday == null ? "" : r.ActualToday.toFixed(2);
    tr.appendChild(tdA);

    const tdP2 = document.createElement("td");
    tdP2.className = "num";
    tdP2.textContent = r.PredictionTomorrow == null ? "" : r.PredictionTomorrow.toFixed(2);
    tr.appendChild(tdP2);

    tbody.appendChild(tr);
  }

  console.log("Rendered table rows:", tbody.children.length);
}

function updateMeta(rows) {
  const meta = $("meta");
  if (!meta) return;
  if (!rows || rows.length === 0) {
    meta.textContent = "";
    return;
  }
  meta.textContent = ` (${rows.length} řádků)`;
}

function render() {
  const { tableRows, chartRows } = applyFilters(allRows);

  // Render table even if chart fails
  renderTable(tableRows);

  try {
    renderChart(chartRows);
  } catch (e) {
    console.error("Chart render failed:", e);
  }

  updateMeta(allRows);
}

async function reload() {
  const csvText = await fetchCSV();
  allRows = parseCSV(csvText);

  console.log("Loaded rows:", allRows); // you already see good data here
  render();
}

function hookUI() {
  $("reloadBtn")?.addEventListener("click", () => reload().catch(console.error));
  $("daysSelect")?.addEventListener("change", () => render());
  $("descToggle")?.addEventListener("change", () => render());
  $("searchInput")?.addEventListener("input", () => render());
}

document.addEventListener("DOMContentLoaded", () => {
  hookUI();
  reload().catch((e) => {
    console.error(e);
    const meta = $("meta");
    if (meta) meta.textContent = ` (chyba: ${e.message})`;
  });
});
