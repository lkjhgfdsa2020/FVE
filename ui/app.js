/* ui/app.js
 * Dashboard for:
 *  - forecasts/forecast_daily_summary.csv
 *  - forecasts/intraday/forecast_intraday_YYYY-MM-DD.csv (today)
 *
 * index.html IDs:
 * - canvas#chart                (daily chart: PredictionToday + ActualToday)
 * - canvas#chartIntraday        (intraday chart: pv_kw_pred today)
 * - tbody#tbody
 * - select#daysSelect
 * - button#reloadBtn
 * - input#searchInput
 * - input#descToggle
 * - span#meta
 * - p#intradayStatus
 */

let dailyChart = null;
let intradayChart = null;
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

function pragueTodayISO() {
  // format YYYY-MM-DD in Europe/Prague
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Prague",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return fmt.format(new Date());
}

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Fetch failed: HTTP ${r.status} (${url})`);
  return await r.text();
}

async function fetchDailyCSV() {
  const url = new URL("../forecasts/forecast_daily_summary.csv", window.location.href);
  url.searchParams.set("_", String(Date.now())); // cache-bust
  return await fetchText(url.toString());
}

async function fetchIntradayCSV(dayISO) {
  const url = new URL(`../forecasts/intraday/forecast_intraday_${dayISO}.csv`, window.location.href);
  url.searchParams.set("_", String(Date.now())); // cache-bust
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
    console.warn("Daily CSV header:", header);
    throw new Error("Daily CSV missing required columns: Date, PredictionToday, PredictionTomorrow");
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
  // expected columns: time,pv_kw_pred,step_kwh,irr_wm2,cloud_cover,pr_used,irr_source
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
  const iIrr = idx("irr_wm2");
  const iCloud = idx("cloud_cover");
  const iPR = idx("pr_used");
  const iSrc = idx("irr_source");

  if (iTime === -1 || iPv === -1) {
    console.warn("Intraday CSV header:", header);
    throw new Error("Intraday CSV missing required columns: time, pv_kw_pred");
  }

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const t = (cols[iTime] ?? "").trim();
    if (!t) continue;

    // Expect "YYYY-MM-DD HH:MM:SS"
    const hhmm = t.length >= 16 ? t.slice(11, 16) : t;

    rows.push({
      time: t,
      hhmm,
      pv_kw_pred: parseNumber(cols[iPv]) ?? 0,
      irr_wm2: iIrr !== -1 ? parseNumber(cols[iIrr]) : null,
      cloud_cover: iCloud !== -1 ? parseNumber(cols[iCloud]) : null,
      pr_used: iPR !== -1 ? parseNumber(cols[iPR]) : null,
      irr_source: iSrc !== -1 ? String(cols[iSrc] ?? "").trim() : "",
    });
  }

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

function ensureFixedChartHeight(canvasId, h = 420) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const wrap = canvas.parentElement;
  if (wrap) {
    wrap.style.height = `${h}px`;
    wrap.style.minHeight = `${h}px`;
    wrap.style.maxHeight = `${h}px`;
  }
  canvas.style.height = `${h}px`;
}

function destroyChartIfAny(canvas, refObjKey) {
  if (!canvas || typeof Chart === "undefined") return;
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
}

function renderDailyChart(rows) {
  const canvas = $("chart");
  if (!canvas) return;

  ensureFixedChartHeight("chart", 420);

  if (typeof Chart === "undefined") {
    throw new Error("Chart.js is not loaded (Chart is undefined).");
  }

  destroyChartIfAny(canvas);
  if (dailyChart) {
    try { dailyChart.destroy(); } catch (_) {}
    dailyChart = null;
  }

  const labels = rows.map((r) => r.Date);
  const predToday = rows.map((r) => r.PredictionToday);
  const actualToday = rows.map((r) => r.ActualToday);

  const suggestedMax = computeSuggestedMax([...predToday, ...actualToday]);

  dailyChart = new Chart(canvas.getContext("2d"), {
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

function renderIntradayChart(intraRows) {
  const canvas = $("chartIntraday");
  if (!canvas) return;

  ensureFixedChartHeight("chartIntraday", 360);

  if (typeof Chart === "undefined") {
    throw new Error("Chart.js is not loaded (Chart is undefined).");
  }

  destroyChartIfAny(canvas);
  if (intradayChart) {
    try { intradayChart.destroy(); } catch (_) {}
    intradayChart = null;
  }

  const labels = intraRows.map((r) => r.hhmm);
  const pv = intraRows.map((r) => r.pv_kw_pred);

  const suggestedMax = computeSuggestedMax([...pv]);

  intradayChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "PV kW pred (30 min)",
          data: pv,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: { beginAtZero: true, suggestedMax, title: { display: true, text: "kW" } },
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
}

function updateMeta(rows) {
  const meta = $("meta");
  if (!meta) return;
  meta.textContent = rows && rows.length ? ` (${rows.length} řádků)` : "";
}

function render() {
  const { tableRows, chartRows } = applyFilters(allRows);

  renderTable(tableRows);

  try {
    renderDailyChart(chartRows);
  } catch (e) {
    console.error("Daily chart render failed:", e);
  }

  updateMeta(allRows);
}

async function loadIntradayToday() {
  const status = $("intradayStatus");
  const dayISO = pragueTodayISO();
  try {
    const txt = await fetchIntradayCSV(dayISO);
    const rows = parseIntradayCSV(txt);
    if (!rows.length) {
      if (status) status.textContent = `Intraday forecast pro ${dayISO} je prázdný.`;
      return;
    }
    if (status) status.textContent = `Zobrazuji intraday forecast pro ${dayISO} (${rows.length} bodů).`;
    renderIntradayChart(rows);
  } catch (e) {
    console.warn("Intraday load failed:", e);
    if (status) status.textContent = `Intraday forecast pro ${dayISO} není k dispozici (ještě neběžel forecast?).`;
  }
}

async function reload() {
  const csvText = await fetchDailyCSV();
  allRows = parseDailyCSV(csvText);
  console.log("Loaded daily rows:", allRows);

  render();
  await loadIntradayToday();
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
