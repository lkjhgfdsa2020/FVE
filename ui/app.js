/* ui/app.js
 * Forecast dashboard – reads forecasts/forecast_daily_summary.csv
 * Columns expected: Date,PredictionToday,ActualToday,PredictionTomorrow,SwitchOff,SwitchOn
 * (ActualToday can be empty)
 */

let chart = null;

const el = (id) => document.getElementById(id);

function parseNumber(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  if (!s) return null;

  // if someone ever uses comma decimals
  const normalized = s.replace(",", ".");
  const n = Number.parseFloat(normalized);
  return Number.isFinite(n) ? n : null;
}

function parseCSV(text) {
  const lines = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n").filter(l => l.trim().length > 0);
  if (lines.length < 2) return [];

  const header = lines[0].split(",").map(h => h.trim());
  const idx = (name) => header.indexOf(name);

  const iDate = idx("Date");
  const iPredT = idx("PredictionToday");
  const iPredTom = idx("PredictionTomorrow");
  const iAct = idx("ActualToday");

  if (iDate === -1 || iPredT === -1 || iPredTom === -1) {
    console.warn("CSV header:", header);
    throw new Error("CSV missing required columns: Date, PredictionToday, PredictionTomorrow");
  }

  const rows = [];
  for (let k = 1; k < lines.length; k++) {
    const cols = lines[k].split(","); // values in your CSV are simple (no quoted commas)
    const d = (cols[iDate] ?? "").trim();
    if (!d) continue;

    rows.push({
      Date: d,
      PredictionToday: parseNumber(cols[iPredT]),
      PredictionTomorrow: parseNumber(cols[iPredTom]),
      ActualToday: iAct !== -1 ? parseNumber(cols[iAct]) : null,
    });
  }

  // sort ascending by Date
  rows.sort((a, b) => a.Date.localeCompare(b.Date));
  return rows;
}

async function fetchWithFallback(paths) {
  let lastErr = null;
  for (const p of paths) {
    try {
      const r = await fetch(p, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${p}`);
      return await r.text();
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error("Failed to fetch CSV");
}

function sliceLast(rows, n) {
  if (!Number.isFinite(n) || n <= 0) return rows;
  if (rows.length <= n) return rows;
  return rows.slice(rows.length - n);
}

function computeYMax(values) {
  const nums = values.filter(v => Number.isFinite(v));
  if (nums.length === 0) return 5;
  const m = Math.max(...nums);
  // add headroom
  const head = m * 1.15 + 0.5;
  // round up to 0.5 kWh steps
  return Math.ceil(head * 2) / 2;
}

function renderChart(rows) {
  const canvas = el("forecastChart");
  if (!canvas) return;

  // FIXED HEIGHT to prevent "growing chart"
  const wrapper = canvas.parentElement;
  if (wrapper) {
    wrapper.style.height = "420px";
    wrapper.style.maxHeight = "420px";
    wrapper.style.minHeight = "420px";
  }
  canvas.style.height = "420px";

  const labels = rows.map(r => r.Date);
  const predToday = rows.map(r => r.PredictionToday);
  const predTomorrow = rows.map(r => r.PredictionTomorrow);
  const actualToday = rows.map(r => r.ActualToday);

  const yMax = computeYMax([...predToday, ...predTomorrow, ...actualToday]);

  const data = {
    labels,
    datasets: [
      {
        label: "PredictionToday (kWh)",
        data: predToday,
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.25,
      },
      {
        label: "PredictionTomorrow (kWh)",
        data: predTomorrow,
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.25,
      },
      // If you want to show actuals later, keep it enabled:
      {
        label: "ActualToday (kWh)",
        data: actualToday,
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.25,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false, // IMPORTANT for fixed height
    animation: false,
    scales: {
      y: {
        beginAtZero: true,
        suggestedMax: yMax,
        ticks: {
          // show nice numbers
          callback: (v) => v,
        },
      },
      x: {
        ticks: {
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 12,
        },
      },
    },
    plugins: {
      legend: {
        display: true,
      },
      tooltip: {
        intersect: false,
        mode: "index",
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
  };

  if (chart) {
    chart.data = data;
    chart.options = options;
    chart.update();
  } else {
    chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data,
      options,
    });
  }
}

function renderTable(rows) {
  const tbody = el("forecastTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");

    const tdDate = document.createElement("td");
    tdDate.textContent = r.Date;
    tr.appendChild(tdDate);

    const tdP1 = document.createElement("td");
    tdP1.textContent = (r.PredictionToday == null) ? "" : r.PredictionToday.toFixed(2);
    tr.appendChild(tdP1);

    const tdA = document.createElement("td");
    tdA.textContent = (r.ActualToday == null) ? "" : r.ActualToday.toFixed(2);
    tr.appendChild(tdA);

    const tdP2 = document.createElement("td");
    tdP2.textContent = (r.PredictionTomorrow == null) ? "" : r.PredictionTomorrow.toFixed(2);
    tr.appendChild(tdP2);

    tbody.appendChild(tr);
  }
}

async function loadAndRender() {
  const daysSelect = el("daysSelect");
  const nDays = daysSelect ? Number.parseInt(daysSelect.value, 10) : 30;

  // Try multiple paths so it works from /ui/ and from repo root
  const csvText = await fetchWithFallback([
    "../forecasts/forecast_daily_summary.csv",
    "./forecasts/forecast_daily_summary.csv",
    "/forecasts/forecast_daily_summary.csv",
  ]);

  const allRows = parseCSV(csvText);
  const rows = sliceLast(allRows, nDays);

  // Debug: if chart is empty, log what we got
  console.log("Loaded rows:", rows);

  renderChart(rows);
  renderTable(rows);
}

function hookUI() {
  const daysSelect = el("daysSelect");
  if (daysSelect) {
    daysSelect.addEventListener("change", () => loadAndRender().catch(console.error));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  hookUI();
  loadAndRender().catch((e) => {
    console.error(e);
    const msg = el("errorBox");
    if (msg) msg.textContent = `Error: ${e.message}`;
  });
});
