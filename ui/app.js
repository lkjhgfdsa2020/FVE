/* ui/app.js
 * Dashboard for forecasts/forecast_daily_summary.csv
 * Expected columns:
 * Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
 */

let chart = null;

const $ = (id) => document.getElementById(id);

function parseNumber(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  if (!s) return null;

  // support comma decimal just in case
  const n = Number.parseFloat(s.replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

function stripBOM(s) {
  if (!s) return s;
  return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s;
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

async function fetchCSV() {
  // For https://.../FVE/ui/ this resolves to https://.../FVE/forecasts/forecast_daily_summary.csv
  const url = new URL("../forecasts/forecast_daily_summary.csv", window.location.href);

  // cache-bust (GitHub Pages can be aggressive)
  url.searchParams.set("_", String(Date.now()));

  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to fetch CSV: HTTP ${r.status}`);
  return await r.text();
}

function sliceLast(rows, n) {
  if (!Number.isFinite(n) || n <= 0) return rows;
  return rows.length <= n ? rows : rows.slice(rows.length - n);
}

function computeSuggestedMax(values) {
  const nums = values.filter((v) => Number.isFinite(v));
  if (nums.length === 0) return 5;
  const m = Math.max(...nums);
  // add headroom and round nicely
  const head = m * 1.15 + 0.5;
  return Math.ceil(head * 2) / 2; // 0.5 kWh steps
}

function ensureFixedChartHeight() {
  const canvas = $("forecastChart");
  if (!canvas) return;

  const wrap = canvas.parentElement;
  if (wrap) {
    wrap.style.height = "420px";
    wrap.style.minHeight = "420px";
    wrap.style.maxHeight = "420px";
  }
  canvas.style.height = "420px";
}

function renderChart(rows) {
  const canvas = $("forecastChart");
  if (!canvas) return;

  ensureFixedChartHeight();

  const labels = rows.map((r) => r.Date);
  const predToday = rows.map((r) => r.PredictionToday);
  const predTomorrow = rows.map((r) => r.PredictionTomorrow);
  const actualToday = rows.map((r) => r.ActualToday);

  const suggestedMax = computeSuggestedMax([...predToday, ...predTomorrow, ...actualToday]);

  const data = {
    labels,
    datasets: [
      {
        label: "PredictionToday (kWh)",
        data: predToday, // array of numbers/nulls
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
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false, // IMPORTANT (fixed height)
    animation: false,
    scales: {
      y: {
        beginAtZero: true,
        suggestedMax,
      },
      x: {
        ticks: {
          autoSkip: true,
          maxTicksLimit: 12,
          maxRotation: 0,
        },
      },
    },
    plugins: {
      legend: { display: true },
      tooltip: { intersect: false, mode: "index" },
    },
    interaction: { intersect: false, mode: "index" },
  };

  if (chart) {
    chart.data = data;
    chart.options = options;
    chart.update();
  } else {
    // Ensure Chart.js is available
    if (typeof Chart === "undefined") {
      throw new Error("Chart.js is not loaded (Chart is undefined). Check index.html script order.");
    }
    chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data,
      options,
    });
  }
}

function renderTable(rows) {
  const tbody = $("forecastTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");

    const tdDate = document.createElement("td");
    tdDate.textContent = r.Date;
    tr.appendChild(tdDate);

    const tdP1 = document.createElement("td");
    tdP1.textContent = r.PredictionToday == null ? "" : r.PredictionToday.toFixed(2);
    tr.appendChild(tdP1);

    const tdA = document.createElement("td");
    tdA.textContent = r.ActualToday == null ? "" : r.ActualToday.toFixed(2);
    tr.appendChild(tdA);

    const tdP2 = document.createElement("td");
    tdP2.textContent = r.PredictionTomorrow == null ? "" : r.PredictionTomorrow.toFixed(2);
    tr.appendChild(tdP2);

    tbody.appendChild(tr);
  }
}

async function loadAndRender() {
  const nDays = $("daysSelect") ? Number.parseInt($("daysSelect").value, 10) : 30;

  const csvText = await fetchCSV();
  const allRows = parseCSV(csvText);
  const rows = sliceLast(allRows, nDays);

  // Debug – if it still draws empty, this will tell us what numbers came in:
  console.log("forecast rows:", rows);

  renderChart(rows);
  renderTable(rows);
}

document.addEventListener("DOMContentLoaded", () => {
  if ($("daysSelect")) {
    $("daysSelect").addEventListener("change", () => loadAndRender().catch(console.error));
  }
  loadAndRender().catch((e) => {
    console.error(e);
    const box = $("errorBox");
    if (box) box.textContent = `Error: ${e.message}`;
  });
});
