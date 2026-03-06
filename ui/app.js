/* ui/app.js
 * Dashboard for forecasts/forecast_daily_summary.csv + intraday hourly curve.
 */

let chartDaily = null;
let chartIntraday = null;
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

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Fetch failed: ${url} (HTTP ${r.status})`);
  return await r.text();
}

async function fetchDailyCSV() {
  const url = new URL("../forecasts/forecast_daily_summary.csv", window.location.href);
  url.searchParams.set("_", String(Date.now()));
  return await fetchText(url.toString());
}

async function fetchIntradayCSV(yyyy_mm_dd) {
  const url = new URL(`../forecasts/intraday/forecast_intraday_${yyyy_mm_dd}.csv`, window.location.href);
  url.searchParams.set("_", String(Date.now()));
  return await fetchText(url.toString());
}

function parseDailyCSV(text) {
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

function parseIntradayCSV(text) {
  const lines = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter((l) => l.trim().length > 0);

  if (lines.length < 2) return [];

  const header = lines[0].split(",").map((h) => stripBOM(h.trim()));
  const idx = (name) => header.indexOf(name);

  const iTime = idx("time");
  const iPv = idx("pv_kw_pred");

  if (iTime === -1 || iPv === -1) {
    console.warn("Intraday header:", header);
    throw new Error("Intraday CSV missing required columns: time, pv_kw_pred");
  }

  const points = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const t = (cols[iTime] ?? "").trim();
    if (!t) continue;
    points.push({
      time: t,
      pv_kw_pred: parseNumber(cols[iPv]) ?? 0,
    });
  }
  return points;
}

function applyFilters(rows) {
  const search = ($("searchInput")?.value || "").trim();
  const desc = $("descToggle")?.checked ?? true;

  let out = rows;
  if (search) out = out.filter((r) => r.Date.includes(search));

  out = out.slice().sort((a, b) => (desc ? b.Date.localeCompare(a.Date) : a.Date.localeCompare(b.Date)));

  const nDays = $("daysSelect") ? Number.parseInt($("daysSelect").value, 10) : 30;
  if (Number.isFinite(nDays) && nDays > 0 && out.length > nDays) out = out.slice(0, nDays);

  const chartRows = out.slice().sort((a, b) => a.Date.localeCompare(b.Date));
  return { tableRows: out, chartRows };
}

function computeSuggestedMax(values) {
  const nums = values.filter((v) => Number.isFinite(v));
  if (nums.length === 0) return 5;
  const m = Math.max(...nums);
  const head = m * 1.15 + 0.5;
  return Math.ceil(head * 2) / 2;
}

function ensureFixedHeight(canvasId, heightPx) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const wrap = canvas.parentElement;
  if (wrap) {
    wrap.style.height = `${heightPx}px`;
    wrap.style.minHeight = `${heightPx}px`;
    wrap.style.maxHeight = `${heightPx}px`;
  }
  canvas.style.height = `${heightPx}px`;
}

function destroyChart(canvas, instance) {
  if (!canvas || typeof Chart === "undefined") return null;
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
  if (instance) {
    try {
      instance.destroy();
    } catch (_) {}
  }
  return null;
}

function renderDailyChart(rows) {
  const canvas = $("chartDaily");
  if (!canvas) return;
  ensureFixedHeight("chartDaily", 380);

  if (typeof Chart === "undefined") throw new Error("Chart.js not loaded.");

  chartDaily = destroyChart(canvas, chartDaily);

  const labels = rows.map((r) => r.Date);
  const predToday = rows.map((r) => r.PredictionToday);
  const actualToday = rows.map((r) => r.ActualToday);

  const suggestedMax = computeSuggestedMax([...predToday, ...actualToday]);

  chartDaily = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "PredictionToday (kWh)", data: predToday, borderWidth: 2, pointRadius: 3, tension: 0.25, spanGaps: true },
        { label: "ActualToday (kWh)", data: actualToday, borderWidth: 2, pointRadius: 3, tension: 0.25, spanGaps: true },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: { y: { beginAtZero: true, suggestedMax }, x: { ticks: { autoSkip: true, maxTicksLimit: 12, maxRotation: 0 } } },
      plugins: { legend: { display: true }, tooltip: { intersect: false, mode: "index" } },
      interaction: { intersect: false, mode: "index" },
    },
  });
}

function renderIntradayChart(points) {
  const canvas = $("chartIntraday");
  const msg = $("intradayMsg");
  if (!canvas) return;

  ensureFixedHeight("chartIntraday", 320);

  if (!points || points.length === 0) {
    if (msg) msg.textContent = "Intraday forecast pro dnešek není k dispozici.";
    chartIntraday = destroyChart(canvas, chartIntraday);
    return;
  }
  if (msg) msg.textContent = "";

  if (typeof Chart === "undefined") throw new Error("Chart.js not loaded.");

  chartIntraday = destroyChart(canvas, chartIntraday);

  const labels = points.map((p) => p.time.slice(11, 16)); // HH:MM
  const pv = points.map((p) => p.pv_kw_pred);

  const suggestedMax = computeSuggestedMax(pv);

  chartIntraday = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [{ label: "PV pred (kW) – dnes (hourly)", data: pv, borderWidth: 2, pointRadius: 2, tension: 0.25, spanGaps: true }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: { y: { beginAtZero: true, suggestedMax }, x: { ticks: { autoSkip: true, maxTicksLimit: 12, maxRotation: 0 } } },
      plugins: { legend: { display: true }, tooltip: { intersect: false, mode: "index" } },
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
}

function updateMeta(rows) {
  const meta = $("meta");
  if (!meta) return;
  meta.textContent = rows && rows.length ? ` (${rows.length} řádků)` : "";
}

function todayPragueISO() {
  const fmt = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Prague", year: "numeric", month: "2-digit", day: "2-digit" });
  return fmt.format(new Date()); // YYYY-MM-DD
}

function render() {
  const { tableRows, chartRows } = applyFilters(allRows);
  renderTable(tableRows);
  try {
    renderDailyChart(chartRows);
  } catch (e) {
    console.error("Daily chart failed:", e);
  }
  updateMeta(allRows);
}

async function reload() {
  const dailyText = await fetchDailyCSV();
  allRows = parseDailyCSV(dailyText);
  console.log("Loaded daily rows:", allRows);
  render();

  const iso = todayPragueISO();
  try {
    const intradayText = await fetchIntradayCSV(iso);
    const points = parseIntradayCSV(intradayText);
    console.log("Loaded intraday points:", points.length);
    renderIntradayChart(points);
  } catch (e) {
    console.warn("Intraday not available:", e);
    renderIntradayChart([]);
  }
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
